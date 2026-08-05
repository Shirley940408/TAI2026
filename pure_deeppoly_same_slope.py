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
            print(f'[dbg]{(" " + tag) if tag else ""} [same-slope] {msg}', flush=True)

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

                    # CROWN-style: use the SAME slope (u/(u-l)) for the lower
                    # bound too, instead of the 0/1 area-minimizing heuristic.
                    # y >= slope_up * z is a valid lower bound for slope_up in
                    # [0,1] (same soundness argument as any alpha in [0,1]).
                    new_A_low[cross_mask] = slope_up.unsqueeze(1) * A_low[cross_mask]
                    new_b_low[cross_mask] = slope_up * b_low[cross_mask]

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

# (TAIvenv310) (base) shuoyan@u172-016-131-162 code % python run_diagnostics_same_slope.py --tests-dir ../test_cases --gt ../test_cases/gt.txt
# \[dbg] fc1/img0_0.09500.txt [same-slope] unstable_relu_neurons=20 min_margin=0.7154 result=VERIFIED (0.003s)
# fc1/img0_0.09500.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc1/img1_0.02000.txt [same-slope] unstable_relu_neurons=11 min_margin=3.8228 result=VERIFIED (0.003s)
# fc1/img1_0.02000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc1/img2_0.11500.txt [same-slope] unstable_relu_neurons=35 min_margin=-6.8011 result=NOT VERIFIED (0.003s)
# fc1/img2_0.11500.txt    gt=verified     pred=not verified       MISMATCH        0.00s
# [dbg] fc1/img3_0.09000.txt [same-slope] unstable_relu_neurons=23 min_margin=0.4138 result=VERIFIED (0.002s)
# fc1/img3_0.09000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc1/img4_0.14000.txt [same-slope] unstable_relu_neurons=43 min_margin=-17.6602 result=NOT VERIFIED (0.003s)
# fc1/img4_0.14000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc2/img0_0.08500.txt [same-slope] unstable_relu_neurons=83 min_margin=-17.8499 result=NOT VERIFIED (0.004s)
# fc2/img0_0.08500.txt    gt=verified     pred=not verified       MISMATCH        0.00s
# [dbg] fc2/img1_0.08000.txt [same-slope] unstable_relu_neurons=76 min_margin=-12.4570 result=NOT VERIFIED (0.004s)
# fc2/img1_0.08000.txt    gt=verified     pred=not verified       MISMATCH        0.00s
# [dbg] fc2/img2_0.17000.txt [same-slope] unstable_relu_neurons=119 min_margin=-82.4940 result=NOT VERIFIED (0.004s)
# fc2/img2_0.17000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc2/img3_0.01000.txt [same-slope] unstable_relu_neurons=14 min_margin=7.5683 result=VERIFIED (0.004s)
# fc2/img3_0.01000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc2/img4_0.01000.txt [same-slope] unstable_relu_neurons=10 min_margin=13.5317 result=VERIFIED (0.004s)
# fc2/img4_0.01000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc3/img0_0.04000.txt [same-slope] unstable_relu_neurons=51 min_margin=-2.4200 result=NOT VERIFIED (0.004s)
# fc3/img0_0.04000.txt    gt=verified     pred=not verified       MISMATCH        0.00s
# [dbg] fc3/img1_0.04500.txt [same-slope] unstable_relu_neurons=33 min_margin=-0.2166 result=NOT VERIFIED (0.004s)
# fc3/img1_0.04500.txt    gt=verified     pred=not verified       MISMATCH        0.00s
# [dbg] fc3/img2_0.04500.txt [same-slope] unstable_relu_neurons=40 min_margin=-2.6321 result=NOT VERIFIED (0.004s)
# fc3/img2_0.04500.txt    gt=verified     pred=not verified       MISMATCH        0.00s
# [dbg] fc3/img3_0.05000.txt [same-slope] unstable_relu_neurons=28 min_margin=-3.6208 result=NOT VERIFIED (0.004s)
# fc3/img3_0.05000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc3/img4_0.02000.txt [same-slope] unstable_relu_neurons=8 min_margin=6.8636 result=VERIFIED (0.004s)
# fc3/img4_0.02000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc4/img0_0.17000.txt [same-slope] unstable_relu_neurons=91 min_margin=-114.8114 result=NOT VERIFIED (0.005s)
# fc4/img0_0.17000.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc4/img1_0.22000.txt [same-slope] unstable_relu_neurons=210 min_margin=-1238.1344 result=NOT VERIFIED (0.005s)
# fc4/img1_0.22000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc4/img2_0.13000.txt [same-slope] unstable_relu_neurons=19 min_margin=0.2188 result=VERIFIED (0.005s)
# fc4/img2_0.13000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc4/img3_0.03500.txt [same-slope] unstable_relu_neurons=9 min_margin=-1.0230 result=NOT VERIFIED (0.004s)
# fc4/img3_0.03500.txt    gt=verified     pred=not verified       MISMATCH        0.00s
# [dbg] fc4/img4_0.16000.txt [same-slope] unstable_relu_neurons=123 min_margin=-162.8233 result=NOT VERIFIED (0.005s)
# fc4/img4_0.16000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc5/img0_0.01000.txt [same-slope] unstable_relu_neurons=6 min_margin=5.3661 result=VERIFIED (0.004s)
# fc5/img0_0.01000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc5/img1_0.12000.txt [same-slope] unstable_relu_neurons=107 min_margin=-11.1498 result=NOT VERIFIED (0.005s)
# fc5/img1_0.12000.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc5/img2_0.01000.txt [same-slope] unstable_relu_neurons=8 min_margin=7.0869 result=VERIFIED (0.004s)
# fc5/img2_0.01000.txt    gt=verified     pred=verified   OK      0.00s
# [dbg] fc5/img3_0.10000.txt [same-slope] unstable_relu_neurons=98 min_margin=-7.7139 result=NOT VERIFIED (0.007s)
# fc5/img3_0.10000.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc5/img4_0.14500.txt [same-slope] unstable_relu_neurons=126 min_margin=-11.6643 result=NOT VERIFIED (0.005s)
# fc5/img4_0.14500.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc6/img0_0.01000.txt [same-slope] unstable_relu_neurons=16 min_margin=17.9075 result=VERIFIED (0.005s)
# fc6/img0_0.01000.txt    gt=verified     pred=verified   OK      0.01s
# [dbg] fc6/img1_0.01000.txt [same-slope] unstable_relu_neurons=18 min_margin=10.9788 result=VERIFIED (0.006s)
# fc6/img1_0.01000.txt    gt=verified     pred=verified   OK      0.01s
# [dbg] fc6/img2_0.07000.txt [same-slope] unstable_relu_neurons=162 min_margin=-77.0497 result=NOT VERIFIED (0.007s)
# fc6/img2_0.07000.txt    gt=not verified pred=not verified       OK      0.01s
# [dbg] fc6/img3_0.08500.txt [same-slope] unstable_relu_neurons=88 min_margin=-11.8427 result=NOT VERIFIED (0.006s)
# fc6/img3_0.08500.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc6/img4_0.05000.txt [same-slope] unstable_relu_neurons=57 min_margin=-2.5350 result=NOT VERIFIED (0.006s)
# fc6/img4_0.05000.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc7/img0_0.50000.txt [same-slope] unstable_relu_neurons=209 min_margin=-430.8070 result=NOT VERIFIED (0.008s)
# fc7/img0_0.50000.txt    gt=not verified pred=not verified       OK      0.01s
# [dbg] fc7/img1_0.14500.txt [same-slope] unstable_relu_neurons=86 min_margin=-8.2350 result=NOT VERIFIED (0.007s)
# fc7/img1_0.14500.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc7/img2_0.17000.txt [same-slope] unstable_relu_neurons=107 min_margin=-12.9356 result=NOT VERIFIED (0.007s)
# fc7/img2_0.17000.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] fc7/img3_0.02000.txt [same-slope] unstable_relu_neurons=29 min_margin=0.8281 result=VERIFIED (0.007s)
# fc7/img3_0.02000.txt    gt=verified     pred=verified   OK      0.01s
# [dbg] fc7/img4_0.15500.txt [same-slope] unstable_relu_neurons=98 min_margin=-12.3919 result=NOT VERIFIED (0.007s)
# fc7/img4_0.15500.txt    gt=verified     pred=not verified       MISMATCH        0.01s
# [dbg] conv1/img0_0.13500.txt [same-slope] unstable_relu_neurons=2129 min_margin=-11.2842 result=NOT VERIFIED (0.165s)
# conv1/img0_0.13500.txt  gt=verified     pred=not verified       MISMATCH        0.16s
# [dbg] conv1/img1_0.13000.txt [same-slope] unstable_relu_neurons=2003 min_margin=-4.3845 result=NOT VERIFIED (0.109s)
# conv1/img1_0.13000.txt  gt=verified     pred=not verified       MISMATCH        0.11s
# [dbg] conv1/img2_0.31000.txt [same-slope] unstable_relu_neurons=2638 min_margin=-201.4156 result=NOT VERIFIED (0.149s)
# conv1/img2_0.31000.txt  gt=not verified pred=not verified       OK      0.15s
# [dbg] conv1/img3_0.04000.txt [same-slope] unstable_relu_neurons=104 min_margin=8.3779 result=VERIFIED (0.141s)
# conv1/img3_0.04000.txt  gt=verified     pred=verified   OK      0.14s
# [dbg] conv1/img4_0.03000.txt [same-slope] unstable_relu_neurons=110 min_margin=8.4729 result=VERIFIED (0.138s)
# conv1/img4_0.03000.txt  gt=verified     pred=verified   OK      0.14s
# [dbg] conv2/img0_0.13000.txt [same-slope] unstable_relu_neurons=267 min_margin=-16.4680 result=NOT VERIFIED (0.655s)
# conv2/img0_0.13000.txt  gt=verified     pred=not verified       MISMATCH        0.66s
# [dbg] conv2/img1_0.01000.txt [same-slope] unstable_relu_neurons=19 min_margin=7.4806 result=VERIFIED (0.656s)
# conv2/img1_0.01000.txt  gt=verified     pred=verified   OK      0.66s
# [dbg] conv2/img2_0.16500.txt [same-slope] unstable_relu_neurons=1769 min_margin=-26.0215 result=NOT VERIFIED (0.664s)
# conv2/img2_0.16500.txt  gt=verified     pred=not verified       MISMATCH        0.66s
# [dbg] conv2/img3_0.27000.txt [same-slope] unstable_relu_neurons=2424 min_margin=-399.4330 result=NOT VERIFIED (0.643s)
# conv2/img3_0.27000.txt  gt=not verified pred=not verified       OK      0.64s
# [dbg] conv2/img4_0.16000.txt [same-slope] unstable_relu_neurons=847 min_margin=-14.5320 result=NOT VERIFIED (0.657s)
# conv2/img4_0.16000.txt  gt=verified     pred=not verified       MISMATCH        0.66s
# [dbg] conv3/img0_0.01000.txt [same-slope] unstable_relu_neurons=17 min_margin=9.0746 result=VERIFIED (0.768s)
# conv3/img0_0.01000.txt  gt=verified     pred=verified   OK      0.77s
# [dbg] conv3/img1_0.18500.txt [same-slope] unstable_relu_neurons=455 min_margin=-48.8785 result=NOT VERIFIED (0.757s)
# conv3/img1_0.18500.txt  gt=verified     pred=not verified       MISMATCH        0.76s
# [dbg] conv3/img2_0.23500.txt [same-slope] unstable_relu_neurons=448 min_margin=-94.1490 result=NOT VERIFIED (0.777s)
# conv3/img2_0.23500.txt  gt=verified     pred=not verified       MISMATCH        0.78s
# [dbg] conv3/img3_0.24000.txt [same-slope] unstable_relu_neurons=601 min_margin=-102.1385 result=NOT VERIFIED (0.771s)
# conv3/img3_0.24000.txt  gt=verified     pred=not verified       MISMATCH        0.77s
# [dbg] conv3/img4_0.24000.txt [same-slope] unstable_relu_neurons=360 min_margin=-84.6498 result=NOT VERIFIED (0.762s)
# conv3/img4_0.24000.txt  gt=verified     pred=not verified       MISMATCH        0.76s

# ==================== SUMMARY ====================
# Overall: 25/50 = 50.0%
# Total wall time: 8.0s, avg 0.16s/case

# net      correct  total 
# conv1           3      5
# conv2           2      5
# conv3           1      5
# fc1             4      5
# fc2             3      5
# fc3             2      5
# fc4             3      5
# fc5             2      5
# fc6             3      5
# fc7             2      5

# No soundness violations (good -- verifier only ever loses points by being imprecise, never by being unsound).

# False negatives (gt=verified, we said not verified) -- 25 cases:
#   fc1/img2_0.11500.txt  (0.00s)
#   fc2/img0_0.08500.txt  (0.00s)
#   fc2/img1_0.08000.txt  (0.00s)
#   fc3/img0_0.04000.txt  (0.00s)
#   fc3/img1_0.04500.txt  (0.00s)
#   fc3/img2_0.04500.txt  (0.00s)
#   fc4/img0_0.17000.txt  (0.01s)
#   fc4/img3_0.03500.txt  (0.00s)
#   fc5/img1_0.12000.txt  (0.01s)
#   fc5/img3_0.10000.txt  (0.01s)
#   fc5/img4_0.14500.txt  (0.01s)
#   fc6/img3_0.08500.txt  (0.01s)
#   fc6/img4_0.05000.txt  (0.01s)
#   fc7/img1_0.14500.txt  (0.01s)
#   fc7/img2_0.17000.txt  (0.01s)
#   fc7/img4_0.15500.txt  (0.01s)
#   conv1/img0_0.13500.txt  (0.16s)
#   conv1/img1_0.13000.txt  (0.11s)
#   conv2/img0_0.13000.txt  (0.66s)
#   conv2/img2_0.16500.txt  (0.66s)
#   conv2/img4_0.16000.txt  (0.66s)
#   conv3/img1_0.18500.txt  (0.76s)
#   conv3/img2_0.23500.txt  (0.78s)
#   conv3/img3_0.24000.txt  (0.77s)
#   conv3/img4_0.24000.txt  (0.76s)
# (TAIvenv310) (base) shuoyan@u172-016-131-162 code % \main()
# function> python case_deep_trace.py --net fc6 --spec ../test_cases/fc6/img3_0.08500.txt --target-label 1