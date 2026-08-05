import argparse
import os
import time

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']


def analyze(model, x, eps, true_label, debug=False, tag=""):
    """
    Minimal, literal fixed-slope DeepPoly: pure affine bound propagation
    (coefficients maintained relative to the original input at every layer,
    which is equivalent to full back-substitution to the input), a single
    fixed 0/1 ReLU lower-slope heuristic (smaller-relaxation-area choice),
    and NO box/IBP intersection anywhere, and NO gradient optimization.

    This is deliberately as close as possible to "the simplest thing that
    could be called DeepPoly" so its pass rate is directly comparable to a
    classmate's plain-DeepPoly baseline. If this scores much higher than the
    hybrid IBP+affine verifier's own "0/1 heuristic alone" checks, the box
    intersection design is the problem. If it scores similarly low, the bug
    is somewhere more fundamental (Linear/Conv2d propagation itself).
    """
    CERT_TOL = 1e-8

    def dbg(msg):
        if debug:
            print(f'[dbg]{(" " + tag) if tag else ""} [pure] {msg}', flush=True)

    start_time = time.perf_counter()

    model.eval()
    device = x.device
    dtype = x.dtype

    layers = list(model.layers.children())
    if not layers or not isinstance(layers[-1], torch.nn.Linear):
        raise NotImplementedError('Expects the final network layer to be Linear.')

    x_detached = x.detach()
    x_flat = x_detached.reshape(-1)
    input_dim = x_flat.numel()

    x_l = torch.clamp(x_flat - eps, 0.0, 1.0)
    x_u = torch.clamp(x_flat + eps, 0.0, 1.0)

    A_low = torch.eye(input_dim, device=device, dtype=dtype)
    A_high = torch.eye(input_dim, device=device, dtype=dtype)
    b_low = torch.zeros(input_dim, device=device, dtype=dtype)
    b_high = torch.zeros(input_dim, device=device, dtype=dtype)

    float_eps = torch.finfo(dtype).eps
    cur_shape = tuple(x_detached.shape[-3:]) if x_detached.dim() >= 3 else None

    def concrete_bounds(A_l, b_l, A_h, b_h):
        A_l_pos, A_l_neg = A_l.clamp(min=0), A_l.clamp(max=0)
        A_h_pos, A_h_neg = A_h.clamp(min=0), A_h.clamp(max=0)
        low = A_l_pos @ x_l + A_l_neg @ x_u + b_l
        high = A_h_pos @ x_u + A_h_neg @ x_l + b_h
        return low, high

    unstable_count = 0

    with torch.no_grad():
        for layer in layers[:-1]:
            if isinstance(layer, Normalization):
                mean = layer.mean.detach().to(device=device, dtype=dtype).reshape(-1)
                sigma = layer.sigma.detach().to(device=device, dtype=dtype).reshape(-1)
                x_l = (x_l - mean) / sigma
                x_u = (x_u - mean) / sigma
                continue

            if isinstance(layer, torch.nn.Flatten):
                continue

            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                new_A_low = W_pos @ A_low + W_neg @ A_high
                new_A_high = W_pos @ A_high + W_neg @ A_low
                new_b_low = W_pos @ b_low + W_neg @ b_high
                new_b_high = W_pos @ b_high + W_neg @ b_low
                if b is not None:
                    new_b_low = new_b_low + b
                    new_b_high = new_b_high + b
                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Conv2d):
                C_in, H_in, W_in = cur_shape
                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                def conv_c(weight, A):
                    A_img = A.t().reshape(input_dim, C_in, H_in, W_in)
                    return F.conv2d(A_img, weight, None, stride=layer.stride,
                                     padding=layer.padding, dilation=layer.dilation, groups=layer.groups)

                low_img = conv_c(W_pos, A_low) + conv_c(W_neg, A_high)
                high_img = conv_c(W_pos, A_high) + conv_c(W_neg, A_low)
                C_out, H_out, W_out = low_img.shape[1:]
                new_A_low = low_img.reshape(input_dim, -1).t()
                new_A_high = high_img.reshape(input_dim, -1).t()

                def conv_b(weight, bvec):
                    b_img = bvec.reshape(1, C_in, H_in, W_in)
                    return F.conv2d(b_img, weight, None, stride=layer.stride,
                                     padding=layer.padding, dilation=layer.dilation, groups=layer.groups).reshape(-1)

                new_b_low = conv_b(W_pos, b_low) + conv_b(W_neg, b_high)
                new_b_high = conv_b(W_pos, b_high) + conv_b(W_neg, b_low)
                if b is not None:
                    bias_flat = b.view(-1, 1, 1).expand(C_out, H_out, W_out).reshape(-1)
                    new_b_low = new_b_low + bias_flat
                    new_b_high = new_b_high + bias_flat
                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high
                cur_shape = (C_out, H_out, W_out)
                continue

            if isinstance(layer, torch.nn.ReLU):
                l, u = concrete_bounds(A_low, b_low, A_high, b_high)

                new_A_low, new_A_high = A_low.clone(), A_high.clone()
                new_b_low, new_b_high = b_low.clone(), b_high.clone()

                neg_mask = u <= 0
                new_A_low[neg_mask] = 0
                new_A_high[neg_mask] = 0
                new_b_low[neg_mask] = 0
                new_b_high[neg_mask] = 0

                cross_mask = (l < 0) & (u > 0)
                unstable_count += int(cross_mask.sum().item())
                if cross_mask.any():
                    l_c, u_c = l[cross_mask], u[cross_mask]
                    slope_up = u_c / (u_c - l_c).clamp_min(float_eps)
                    new_A_high[cross_mask] = slope_up.unsqueeze(1) * A_high[cross_mask]
                    new_b_high[cross_mask] = slope_up * b_high[cross_mask] - slope_up * l_c

                    # fixed 0/1 heuristic: pick whichever line covers less area
                    use_identity = u_c >= -l_c
                    keep_idx = cross_mask.clone()
                    keep_idx[cross_mask] = ~use_identity  # entries that get slope=0
                    new_A_low[keep_idx] = 0
                    new_b_low[keep_idx] = 0
                    # entries with use_identity=True keep slope=1 (already cloned, unchanged)

                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high
                continue

            raise NotImplementedError('Unsupported layer type: {}'.format(type(layer)))

        # final Linear layer: direct margin bound, same last-layer tightening trick
        output_layer = layers[-1]
        W_out = output_layer.weight.detach().to(device=device, dtype=dtype)
        b_out = output_layer.bias
        b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype) if b_out is None else b_out.detach().to(device=device, dtype=dtype)

        other_labels = [label for label in range(W_out.size(0)) if label != true_label]
        certified_all = True
        min_margin = float('inf')
        for target_label in other_labels:
            w_margin = W_out[true_label] - W_out[target_label]
            b_margin = b_out[true_label] - b_out[target_label]
            w_pos, w_neg = w_margin.clamp(min=0), w_margin.clamp(max=0)
            margin_A_low = w_pos @ A_low + w_neg @ A_high
            margin_b_low = w_pos @ b_low + w_neg @ b_high + b_margin
            A_pos, A_neg = margin_A_low.clamp(min=0), margin_A_low.clamp(max=0)
            margin_low = (A_pos @ x_l + A_neg @ x_u + margin_b_low).item()
            min_margin = min(min_margin, margin_low)
            if margin_low <= CERT_TOL:
                certified_all = False

        dbg(f'unstable_relu_neurons={unstable_count} min_margin={min_margin:.4f} result={"VERIFIED" if certified_all else "NOT VERIFIED"} ({time.perf_counter()-start_time:.3f}s)')
        return certified_all


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


def run_single_case(net_name, spec_path, debug=False, tag=""):
    true_label, pixel_values, eps = parse_spec(spec_path)
    net = load_network(net_name)
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label
    return analyze(net, inputs, eps, true_label, debug=debug, tag=tag)


def run_all_cases(test_dir):
    base_dir = os.path.abspath(test_dir)
    for net_name in NETWORK_NAMES:
        net_dir = os.path.join(base_dir, net_name)
        if not os.path.isdir(net_dir):
            print(f'[skip] {net_name}: missing directory {net_dir}')
            continue
        spec_paths = sorted(
            os.path.join(net_dir, filename) for filename in os.listdir(net_dir) if filename.endswith('.txt')
        )
        for spec_path in spec_paths:
            try:
                result = run_single_case(net_name, spec_path)
            except Exception as exc:
                print(f'{net_name}\t{os.path.basename(spec_path)}\tERROR\t{exc}')
                continue
            status = 'verified' if result else 'not verified'
            print(f'{net_name}\t{os.path.basename(spec_path)}\t{status}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, choices=NETWORK_NAMES)
    parser.add_argument('--spec', type=str)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--tests-dir', type=str, default='../test_cases')
    parser.add_argument('--debug', action='store_true')
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

    print('verified' if analyze(net, inputs, eps, true_label, debug=args.debug, tag=os.path.basename(args.spec)) else 'not verified')


if __name__ == '__main__':
    main()