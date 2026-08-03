import argparse
import os

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']



def analyze(model, x, eps, true_label):
    """
    Hybrid IBP + affine-bound verifier with optimized ReLU slopes.

    Main changes compared with the previous version:
      1. Keep both symbolic affine bounds and concrete IBP bounds.
      2. At every ReLU, intersect the two pre-activation intervals before
         constructing the relaxation. This keeps the useful fact ReLU(z) >= 0.
      3. Replace separate output-logit bounds with direct pairwise margins:
             f_true(x) - f_other(x) > 0.
      4. First optimize one shared set of slopes; then optimize a separate set
         for each still-unproved target class. Different sound relaxations may
         be used to prove different pairwise inequalities.

    For every unstable ReLU l < 0 < u, the symbolic lower relaxation is

        ReLU(z) >= alpha * z,   alpha in [0, 1].

    All computation is CPU-compatible and uses only PyTorch.
    """
    # Keep these modest for the 3-minute per-case limit. Increase them only
    # after measuring the slowest convolutional case on your machine.
    SHARED_OPT_STEPS = 18
    TARGET_OPT_STEPS = 12
    OPT_LR = 5e-2
    OPT_TEMPERATURE = 0.5
    OPT_INIT_EPS = 5e-2
    CERT_TOL = 1e-8

    model.eval()
    device = x.device
    dtype = x.dtype

    layers = list(model.layers.children())
    if not layers or not isinstance(layers[-1], torch.nn.Linear):
        raise NotImplementedError(
            'This implementation expects the final network layer to be Linear.'
        )

    body_layers = layers[:-1]
    output_layer = layers[-1]

    x_detached = x.detach()
    x_flat = x_detached.reshape(-1)
    input_dim = x_flat.numel()

    # The project perturbation set is also clipped to the valid image range.
    original_x_l = torch.clamp(x_flat - eps, 0.0, 1.0)
    original_x_u = torch.clamp(x_flat + eps, 0.0, 1.0)
    original_box_l = original_x_l.reshape_as(x_detached)
    original_box_u = original_x_u.reshape_as(x_detached)

    other_labels = torch.tensor(
        [label for label in range(10) if label != true_label],
        device=device,
        dtype=torch.long,
    )

    float_eps = torch.finfo(dtype).eps

    def concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u):
        """Concretize affine lower/upper expressions over [x_l, x_u]."""
        A_low_pos = A_low.clamp(min=0)
        A_low_neg = A_low.clamp(max=0)
        A_high_pos = A_high.clamp(min=0)
        A_high_neg = A_high.clamp(max=0)

        low = A_low_pos @ x_l + A_low_neg @ x_u + b_low
        high = A_high_pos @ x_u + A_high_neg @ x_l + b_high
        return low, high

    def concrete_lower(A, b, x_l, x_u):
        """Concretize only a symbolic lower affine expression."""
        A_pos = A.clamp(min=0)
        A_neg = A.clamp(max=0)
        return A_pos @ x_l + A_neg @ x_u + b

    def make_parameters(alpha_values, move_inside=False):
        """Clone slope tensors as independent learnable parameters."""
        params = []
        for value in alpha_values:
            init = value.detach().clone()
            if move_inside:
                init.clamp_(OPT_INIT_EPS, 1.0 - OPT_INIT_EPS)
            params.append(torch.nn.Parameter(init))
        return params

    def project_alphas(alpha_params):
        with torch.no_grad():
            for alpha in alpha_params:
                alpha.clamp_(0.0, 1.0)

    def propagate(alpha_params=None, initialize_alpha=False, target_positions=None):
        """
        Propagate hybrid bounds and return direct lower bounds on output margins.

        target_positions indexes `other_labels`, not the original class labels.
        When it is None, all nine pairwise margins are returned.
        """
        if alpha_params is None:
            alpha_params = []

        # Symbolic bounds are expressed over the current normalized input box.
        x_l = original_x_l
        x_u = original_x_u

        A_low = torch.eye(input_dim, device=device, dtype=dtype)
        A_high = torch.eye(input_dim, device=device, dtype=dtype)
        b_low = torch.zeros(input_dim, device=device, dtype=dtype)
        b_high = torch.zeros(input_dim, device=device, dtype=dtype)

        # A second, independent IBP state is preserved throughout the network.
        box_low = original_box_l
        box_high = original_box_u

        cur_shape = tuple(x_detached.shape[-3:]) if x_detached.dim() >= 3 else None
        relu_id = 0

        for layer in body_layers:
            if isinstance(layer, Normalization):
                mean_full = layer.mean.detach().to(device=device, dtype=dtype)
                sigma_full = layer.sigma.detach().to(device=device, dtype=dtype)
                if torch.any(sigma_full <= 0):
                    raise ValueError('Normalization sigma must be positive.')

                # Update the IBP tensor state.
                box_low = (box_low - mean_full) / sigma_full
                box_high = (box_high - mean_full) / sigma_full

                # The symbolic identity is now interpreted over the normalized
                # input interval, so only its domain endpoints change.
                mean_flat = mean_full.reshape(-1)
                sigma_flat = sigma_full.reshape(-1)
                x_l = (x_l - mean_flat) / sigma_flat
                x_u = (x_u - mean_flat) / sigma_flat
                continue

            if isinstance(layer, torch.nn.Flatten):
                box_low = box_low.reshape(box_low.size(0), -1)
                box_high = box_high.reshape(box_high.size(0), -1)
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(
                    device=device, dtype=dtype
                )
                W_pos = W.clamp(min=0)
                W_neg = W.clamp(max=0)

                # Symbolic affine propagation.
                new_A_low = W_pos @ A_low + W_neg @ A_high
                new_A_high = W_pos @ A_high + W_neg @ A_low
                new_b_low = W_pos @ b_low + W_neg @ b_high
                new_b_high = W_pos @ b_high + W_neg @ b_low
                if b is not None:
                    new_b_low = new_b_low + b
                    new_b_high = new_b_high + b
                A_low, A_high = new_A_low, new_A_high
                b_low, b_high = new_b_low, new_b_high

                # Parallel IBP propagation.
                box_low_flat = box_low.reshape(box_low.size(0), -1)
                box_high_flat = box_high.reshape(box_high.size(0), -1)
                new_box_low = box_low_flat @ W_pos.t() + box_high_flat @ W_neg.t()
                new_box_high = box_high_flat @ W_pos.t() + box_low_flat @ W_neg.t()
                if b is not None:
                    new_box_low = new_box_low + b
                    new_box_high = new_box_high + b
                box_low, box_high = new_box_low, new_box_high
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Conv2d):
                if cur_shape is None:
                    # Recover the feature-map shape directly from the IBP state.
                    if box_low.dim() != 4:
                        raise RuntimeError(
                            'Conv2d encountered without a valid feature-map shape.'
                        )
                    cur_shape = tuple(box_low.shape[-3:])

                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(
                    device=device, dtype=dtype
                )
                C_in, H_in, W_in = cur_shape
                W_pos = W.clamp(min=0)
                W_neg = W.clamp(max=0)

                def conv_coeff(weight, coeff):
                    # coeff: [N_previous, input_dim]. Treat input_dim as batch.
                    coeff_img = coeff.t().reshape(input_dim, C_in, H_in, W_in)
                    return F.conv2d(
                        coeff_img,
                        weight,
                        bias=None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )

                low_img = conv_coeff(W_pos, A_low) + conv_coeff(W_neg, A_high)
                high_img = conv_coeff(W_pos, A_high) + conv_coeff(W_neg, A_low)
                C_out, H_out, W_out = low_img.shape[1:]
                new_A_low = low_img.reshape(input_dim, -1).t()
                new_A_high = high_img.reshape(input_dim, -1).t()

                def conv_bias(weight, bias_vector):
                    bias_img = bias_vector.reshape(1, C_in, H_in, W_in)
                    return F.conv2d(
                        bias_img,
                        weight,
                        bias=None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    ).reshape(-1)

                new_b_low = conv_bias(W_pos, b_low) + conv_bias(W_neg, b_high)
                new_b_high = conv_bias(W_pos, b_high) + conv_bias(W_neg, b_low)
                if b is not None:
                    expanded_bias = b.view(-1, 1, 1).expand(
                        C_out, H_out, W_out
                    ).reshape(-1)
                    new_b_low = new_b_low + expanded_bias
                    new_b_high = new_b_high + expanded_bias
                A_low, A_high = new_A_low, new_A_high
                b_low, b_high = new_b_low, new_b_high

                # Parallel convolutional IBP propagation.
                new_box_low = F.conv2d(
                    box_low,
                    W_pos,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                ) + F.conv2d(
                    box_high,
                    W_neg,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                new_box_high = F.conv2d(
                    box_high,
                    W_pos,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                ) + F.conv2d(
                    box_low,
                    W_neg,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                if b is not None:
                    bias_img = b.view(1, -1, 1, 1)
                    new_box_low = new_box_low + bias_img
                    new_box_high = new_box_high + bias_img
                box_low, box_high = new_box_low, new_box_high
                cur_shape = (C_out, H_out, W_out)
                continue

            if isinstance(layer, torch.nn.ReLU):
                affine_l, affine_u = concrete_bounds(
                    A_low, b_low, A_high, b_high, x_l, x_u
                )
                box_l_flat = box_low.reshape(-1)
                box_u_flat = box_high.reshape(-1)

                # Intersect two independently sound abstractions.
                l = torch.maximum(affine_l, box_l_flat)
                u = torch.minimum(affine_u, box_u_flat)

                # Avoid tiny contradictory intervals caused only by float noise.
                u = torch.maximum(u, l)

                neg_mask = u <= 0
                pos_mask = l >= 0
                cross_mask = (~neg_mask) & (~pos_mask)
                neuron_count = l.numel()

                if initialize_alpha:
                    initial_alpha = torch.where(
                        u >= -l,
                        torch.ones_like(l),
                        torch.zeros_like(l),
                    )
                    alpha_params.append(
                        torch.nn.Parameter(initial_alpha.detach().clone())
                    )

                if relu_id >= len(alpha_params):
                    raise RuntimeError(
                        'Missing alpha parameters for ReLU layer {}'.format(relu_id)
                    )
                alpha = alpha_params[relu_id]
                if alpha.numel() != neuron_count:
                    raise RuntimeError(
                        'Alpha size mismatch at ReLU layer {}: expected {}, got {}'.format(
                            relu_id, neuron_count, alpha.numel()
                        )
                    )
                relu_id += 1

                alpha_safe = alpha.clamp(0.0, 1.0)
                zeros = torch.zeros_like(l)
                ones = torch.ones_like(l)

                # Symbolic lower relaxation. The separate box state retains
                # the stronger concrete fact ReLU(z) >= 0.
                lower_scale = torch.where(
                    pos_mask,
                    ones,
                    torch.where(cross_mask, alpha_safe, zeros),
                )
                A_low = lower_scale.unsqueeze(1) * A_low
                b_low = lower_scale * b_low

                # Standard secant upper relaxation over the tightened [l, u].
                denominator = torch.where(
                    cross_mask,
                    (u - l).clamp_min(float_eps),
                    torch.ones_like(u),
                )
                upper_slope_cross = torch.where(
                    cross_mask,
                    u / denominator,
                    torch.zeros_like(u),
                )
                upper_scale = torch.where(
                    pos_mask,
                    ones,
                    torch.where(cross_mask, upper_slope_cross, zeros),
                )
                upper_intercept = torch.where(
                    cross_mask,
                    -upper_slope_cross * l,
                    zeros,
                )
                A_high = upper_scale.unsqueeze(1) * A_high
                b_high = upper_scale * b_high + upper_intercept

                # Keep exact interval non-negativity for all future IBP steps.
                box_low = torch.clamp(l, min=0).reshape_as(box_low)
                box_high = torch.clamp(u, min=0).reshape_as(box_high)
                continue

            raise NotImplementedError(
                'Unsupported layer type before output: {}'.format(type(layer))
            )

        # Directly propagate pairwise margins through the final Linear layer.
        W_out = output_layer.weight.detach().to(device=device, dtype=dtype)
        b_out = output_layer.bias
        if b_out is None:
            b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype)
        else:
            b_out = b_out.detach().to(device=device, dtype=dtype)

        if target_positions is None:
            selected_labels = other_labels
        else:
            selected_labels = other_labels.index_select(0, target_positions)

        W_margin = W_out[true_label].unsqueeze(0) - W_out.index_select(
            0, selected_labels
        )
        b_margin = b_out[true_label] - b_out.index_select(0, selected_labels)
        W_margin_pos = W_margin.clamp(min=0)
        W_margin_neg = W_margin.clamp(max=0)

        margin_A_low = W_margin_pos @ A_low + W_margin_neg @ A_high
        margin_b_low = (
            W_margin_pos @ b_low
            + W_margin_neg @ b_high
            + b_margin
        )
        affine_margin_lower = concrete_lower(
            margin_A_low, margin_b_low, x_l, x_u
        )

        box_l_flat = box_low.reshape(-1)
        box_u_flat = box_high.reshape(-1)
        box_margin_lower = (
            W_margin_pos @ box_l_flat
            + W_margin_neg @ box_u_flat
            + b_margin
        )

        # Either sound abstraction may prove the margin; their maximum remains
        # a sound lower bound because both are lower bounds on the same value.
        hybrid_margin_lower = torch.maximum(
            affine_margin_lower,
            box_margin_lower,
        )
        return hybrid_margin_lower, alpha_params

    # Build the initial 0/1 DeepPoly heuristic and evaluate all margins.
    initial_alpha_params = []
    with torch.enable_grad():
        initial_margins, initial_alpha_params = propagate(
            alpha_params=initial_alpha_params,
            initialize_alpha=True,
        )

    certified = initial_margins.detach() > CERT_TOL
    if bool(certified.all()):
        return True

    if not initial_alpha_params:
        return False

    # Preserve the exact heuristic as a candidate, but optimize from slightly
    # inside the interval so projected Adam is less likely to stick at 0 or 1.
    shared_params = make_parameters(initial_alpha_params, move_inside=True)
    shared_optimizer = torch.optim.Adam(shared_params, lr=OPT_LR)

    with torch.enable_grad():
        for _ in range(SHARED_OPT_STEPS):
            shared_optimizer.zero_grad(set_to_none=True)
            margins, _ = propagate(alpha_params=shared_params)

            # Different iterations may certify different target classes. This
            # is sound: each pairwise inequality may use its own relaxation.
            certified = certified | (margins.detach() > CERT_TOL)
            if bool(certified.all()):
                return True

            unresolved = ~certified
            unresolved_margins = margins[unresolved]
            if unresolved_margins.numel() == 0:
                return True

            loss = OPT_TEMPERATURE * torch.logsumexp(
                -unresolved_margins / OPT_TEMPERATURE,
                dim=0,
            )
            if not torch.isfinite(loss):
                break

            loss.backward()
            shared_optimizer.step()
            project_alphas(shared_params)

    # Check the state after the final shared optimizer update.
    with torch.no_grad():
        shared_margins, _ = propagate(alpha_params=shared_params)
        certified = certified | (shared_margins > CERT_TOL)
        if bool(certified.all()):
            return True

    # Optimize a separate slope set for each remaining target class. A distinct
    # alpha proof for each f_true - f_target inequality is still fully sound.
    unresolved_positions = (~certified).nonzero(as_tuple=True)[0].tolist()

    # Start each target-specific search from the shared result, which usually
    # gives a better warm start than returning to the original 0/1 heuristic.
    shared_values = [alpha.detach().clone() for alpha in shared_params]

    for target_position in unresolved_positions:
        target_index = torch.tensor(
            [target_position], device=device, dtype=torch.long
        )
        target_params = make_parameters(shared_values, move_inside=False)
        target_optimizer = torch.optim.Adam(target_params, lr=OPT_LR)

        target_certified = False
        with torch.enable_grad():
            for _ in range(TARGET_OPT_STEPS):
                target_optimizer.zero_grad(set_to_none=True)
                target_margin, _ = propagate(
                    alpha_params=target_params,
                    target_positions=target_index,
                )
                scalar_margin = target_margin[0]

                if float(scalar_margin.detach()) > CERT_TOL:
                    target_certified = True
                    break

                loss = -scalar_margin
                if not torch.isfinite(loss):
                    break

                loss.backward()
                target_optimizer.step()
                project_alphas(target_params)

        if not target_certified:
            with torch.no_grad():
                final_target_margin, _ = propagate(
                    alpha_params=target_params,
                    target_positions=target_index,
                )
                target_certified = bool(
                    final_target_margin[0].item() > CERT_TOL
                )

        if not target_certified:
            return False

    return True

def load_network(net_name):
    if net_name == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif net_name == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif net_name == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net_name == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif net_name == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        raise ValueError(f'Unsupported net name: {net_name}')

    net.load_state_dict(torch.load(f'../mnist_nets/{net_name}.pt', map_location=torch.device(DEVICE)))
    return net


def parse_spec(spec_path):
    with open(spec_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    true_label = int(lines[0])
    pixel_values = [float(line) for line in lines[1:]]
    eps = float(spec_path[:-4].split('/')[-1].split('_')[-1])
    return true_label, pixel_values, eps


def run_single_case(net_name, spec_path):
    true_label, pixel_values, eps = parse_spec(spec_path)
    net = load_network(net_name)

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    result = analyze(net, inputs, eps, true_label)
    return result


def run_all_cases(test_dir):
    base_dir = os.path.abspath(test_dir)

    for net_name in NETWORK_NAMES:
        net_dir = os.path.join(base_dir, net_name)
        if not os.path.isdir(net_dir):
            print(f'[skip] {net_name}: missing directory {net_dir}')
            continue

        spec_paths = sorted(
            os.path.join(net_dir, filename)
            for filename in os.listdir(net_dir)
            if filename.endswith('.txt')
        )

        for spec_path in spec_paths:
            try:
                result = run_single_case(net_name, spec_path)
            except Exception as exc:
                print(f'{net_name}\t{os.path.basename(spec_path)}\tERROR\t{exc}')
                continue

            status = 'verified' if result else 'not verified'
            print(f'{net_name}\t{os.path.basename(spec_path)}\t{status}')

# run single case: python verifier.py --net fc1 --spec ../test_cases/fc1/img0_0.09500.txt
# run all cases: python verifier.py --batch --tests-dir ../test_cases

def main():
    parser = argparse.ArgumentParser(description='Neural network verification')
    parser.add_argument('--net', type=str, choices=NETWORK_NAMES, help='Neural network architecture')
    parser.add_argument('--spec', type=str, help='Test case to verify')
    parser.add_argument('--batch', action='store_true', help='Run all test cases under the tests directory')
    parser.add_argument('--tests-dir', type=str, default='../test_cases', help='Directory with the test case folders')
    args = parser.parse_args()

    if args.batch:
        run_all_cases(args.tests_dir)
        return

    if args.net is None or args.spec is None:
        parser.error('Please provide both --net and --spec, or use --batch.')

    true_label, pixel_values, eps = parse_spec(args.spec)
    net = load_network(args.net)

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()