"""
Usage (from the same directory as your verifier files):

    python sanity_check_zero_eps.py --tests-dir ../test_cases

For each of the 10 networks, picks one example spec file, forces eps to a
tiny value (1e-9, to avoid any literal division-by-zero edge cases while
still being effectively zero-width), and compares the affine propagation's
computed [low, high] for every output logit against the network's real
forward-pass output at that exact point.

With eps this small, every ReLU is either strictly positive or strictly
negative pre-activation almost surely (measure zero to land exactly on the
boundary) -- so the affine relaxation should be EXACT: low == high == the
real logit, for all 10 classes, up to floating point tolerance. If it's not,
that's a direct, localized proof of a bug in the core propagation, and this
script also prints per-layer intermediate low/high vs. the real intermediate
activation so you can see exactly where the two start to diverge.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']
TOL = 1e-3  # generous float tolerance; real bugs will be off by much more than this


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
    return true_label, pixel_values


def compute_affine_output_bounds(model, x, eps, verbose_layer_check=True):
    """
    Same core Linear/Conv2d/ReLU/Normalization propagation as your other
    verifiers, but returns ALL 10 output logit [low, high] pairs (not just a
    margin), and optionally prints, layer by layer, the widest gap between
    the computed [low, high] and the real intermediate activation at x --
    this pinpoints exactly which layer first introduces unexpected slack.
    """
    device, dtype = x.device, x.dtype
    layers = list(model.layers.children())

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

    # For the layer-by-layer comparison: also run the REAL network forward,
    # step by step, on the exact point x (not perturbed at all).
    real_cur = x_detached.clone()
    layer_idx = 0

    with torch.no_grad():
        for layer in layers[:-1]:
            layer_idx += 1
            if isinstance(layer, Normalization):
                mean = layer.mean.detach().to(device=device, dtype=dtype).reshape(-1)
                sigma = layer.sigma.detach().to(device=device, dtype=dtype).reshape(-1)
                x_l = (x_l - mean) / sigma
                x_u = (x_u - mean) / sigma
                real_cur = layer(real_cur)
                continue

            if isinstance(layer, torch.nn.Flatten):
                real_cur = layer(real_cur)
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
                real_cur = layer(real_cur)
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
                real_cur = layer(real_cur)
                continue

            if isinstance(layer, torch.nn.ReLU):
                l, u = concrete_bounds(A_low, b_low, A_high, b_high)

                if verbose_layer_check:
                    real_flat = real_cur.reshape(-1)
                    gap_low = (real_flat - l).clamp(min=0).max().item()   # real should be >= l
                    gap_high = (u - real_flat).clamp(min=0).min().item()  # not meaningful as min; use max violation instead
                    violation_low = (l - real_flat).clamp(min=0).max().item()   # >0 means l > real (unsound / bug)
                    violation_high = (real_flat - u).clamp(min=0).max().item()  # >0 means real > u (unsound / bug)
                    slack = (u - l).max().item()
                    print(f'    layer {layer_idx} (pre-ReLU): max(l-real)={violation_low:.6g} '
                          f'max(real-u)={violation_high:.6g} max(u-l) slack={slack:.6g}')

                new_A_low, new_A_high = A_low.clone(), A_high.clone()
                new_b_low, new_b_high = b_low.clone(), b_high.clone()
                neg_mask = u <= 0
                new_A_low[neg_mask] = 0
                new_A_high[neg_mask] = 0
                new_b_low[neg_mask] = 0
                new_b_high[neg_mask] = 0
                cross_mask = (l < 0) & (u > 0)
                if cross_mask.any():
                    l_c, u_c = l[cross_mask], u[cross_mask]
                    slope_up = u_c / (u_c - l_c).clamp_min(float_eps)
                    new_A_high[cross_mask] = slope_up.unsqueeze(1) * A_high[cross_mask]
                    new_b_high[cross_mask] = slope_up * b_high[cross_mask] - slope_up * l_c
                    use_identity = u_c >= -l_c
                    keep_idx = cross_mask.clone()
                    keep_idx[cross_mask] = ~use_identity
                    new_A_low[keep_idx] = 0
                    new_b_low[keep_idx] = 0
                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high
                real_cur = layer(real_cur)
                continue

            raise NotImplementedError('Unsupported layer type: {}'.format(type(layer)))

        output_layer = layers[-1]
        W_out = output_layer.weight.detach().to(device=device, dtype=dtype)
        b_out = output_layer.bias
        b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype) if b_out is None else b_out.detach().to(device=device, dtype=dtype)
        W_pos, W_neg = W_out.clamp(min=0), W_out.clamp(max=0)

        out_A_low = W_pos @ A_low + W_neg @ A_high
        out_A_high = W_pos @ A_high + W_neg @ A_low
        out_b_low = W_pos @ b_low + W_neg @ b_high + b_out
        out_b_high = W_pos @ b_high + W_neg @ b_low + b_out
        out_low, out_high = concrete_bounds(out_A_low, out_b_low, out_A_high, out_b_high)

        real_logits = model(x_detached).reshape(-1)

    return out_low, out_high, real_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests-dir', type=str, default='../test_cases')
    args = parser.parse_args()

    print(f'{"net":8s} {"max(l-real)":>14s} {"max(real-u)":>14s} {"verdict"}')
    any_bug = False

    for net_name in NETWORK_NAMES:
        net_dir = os.path.join(args.tests_dir, net_name)
        if not os.path.isdir(net_dir):
            print(f'[skip] {net_name}: missing directory {net_dir}')
            continue
        spec_files = sorted(f for f in os.listdir(net_dir) if f.endswith('.txt'))
        if not spec_files:
            print(f'[skip] {net_name}: no spec files found')
            continue
        spec_path = os.path.join(net_dir, spec_files[0])

        true_label, pixel_values = parse_spec(spec_path)
        net = load_network(net_name)
        x = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

        real_pred = net(x).max(dim=1)[1].item()
        assert real_pred == true_label, f'{net_name}: spec file label mismatch, spec itself is inconsistent'

        print(f'--- {net_name} ({os.path.basename(spec_path)}) ---')
        low, high, real_logits = compute_affine_output_bounds(net, x, eps=1e-9)

        violation_low = (low - real_logits).clamp(min=0).max().item()
        violation_high = (real_logits - high).clamp(min=0).max().item()
        verdict = 'OK' if violation_low < TOL and violation_high < TOL else '*** MISMATCH ***'
        if verdict != 'OK':
            any_bug = True
        print(f'{net_name:8s} {violation_low:14.6g} {violation_high:14.6g} {verdict}')
        print(f'    real logits:  {[round(v, 4) for v in real_logits.tolist()]}')
        print(f'    computed low: {[round(v, 4) for v in low.tolist()]}')
        print(f'    computed high:{[round(v, 4) for v in high.tolist()]}')
        print()

    print('=' * 60)
    if any_bug:
        print('*** At least one network shows a real mismatch at eps~=0. ***')
        print('Look at the first "layer N (pre-ReLU)" line with a nonzero')
        print('violation_low/violation_high above -- that pinpoints where the')
        print('propagation first diverges from the real network.')
    else:
        print('All networks match the real forward pass at eps~=0.')
        print('This rules out a systematic bug in the core Linear/Conv2d/ReLU/')
        print('Normalization propagation itself.')


if __name__ == '__main__':
    main()

# (TAIvenv310) (base) shuoyan@u172-016-131-162 code % python sanity_check_zero_eps.py --tests-dir ../test_cases
# net         max(l-real)    max(real-u) verdict
# --- fc1 (img0_0.09500.txt) ---
#     layer 4 (pre-ReLU): max(l-real)=1.90735e-06 max(real-u)=1.66893e-06 max(u-l) slack=0
# fc1         1.90735e-06     3.8147e-06 OK
#     real logits:  [-17.6036, -8.3817, -7.1495, 11.2083, -6.5147, -1.9944, -25.598, -8.0062, -1.2209, -0.6209]
#     computed low: [-17.6036, -8.3817, -7.1495, 11.2083, -6.5147, -1.9944, -25.598, -8.0062, -1.2209, -0.6209]
#     computed high:[-17.6036, -8.3817, -7.1495, 11.2083, -6.5147, -1.9944, -25.598, -8.0062, -1.2209, -0.6209]

# --- fc2 (img0_0.08500.txt) ---
#     layer 4 (pre-ReLU): max(l-real)=1.90735e-06 max(real-u)=1.43051e-06 max(u-l) slack=0
#     layer 6 (pre-ReLU): max(l-real)=1.90735e-06 max(real-u)=1.90735e-06 max(u-l) slack=0
# fc2         9.53674e-07    1.90735e-06 OK
#     real logits:  [-6.8332, 0.1791, 13.561, -1.5367, -13.7004, -9.1978, -6.4676, -6.4147, -3.0088, -12.9964]
#     computed low: [-6.8332, 0.1791, 13.561, -1.5367, -13.7004, -9.1978, -6.4676, -6.4147, -3.0088, -12.9964]
#     computed high:[-6.8332, 0.1791, 13.561, -1.5367, -13.7004, -9.1978, -6.4676, -6.4147, -3.0088, -12.9964]

# --- fc3 (img0_0.04000.txt) ---
#     layer 4 (pre-ReLU): max(l-real)=4.76837e-07 max(real-u)=4.76837e-07 max(u-l) slack=0
#     layer 6 (pre-ReLU): max(l-real)=4.76837e-07 max(real-u)=4.76837e-07 max(u-l) slack=0
# fc3                   0    1.90735e-06 OK
#     real logits:  [-11.3147, 4.4816, -5.8487, -5.9019, -4.5257, -7.8268, -6.3512, -1.6813, -3.8791, -6.2036]
#     computed low: [-11.3147, 4.4816, -5.8488, -5.9019, -4.5257, -7.8268, -6.3512, -1.6813, -3.8791, -6.2036]
#     computed high:[-11.3147, 4.4816, -5.8488, -5.9019, -4.5257, -7.8268, -6.3512, -1.6813, -3.8791, -6.2036]

# --- fc4 (img0_0.17000.txt) ---
#     layer 4 (pre-ReLU): max(l-real)=3.8147e-06 max(real-u)=7.62939e-06 max(u-l) slack=0
#     layer 6 (pre-ReLU): max(l-real)=1.52588e-05 max(real-u)=1.52588e-05 max(u-l) slack=0
#     layer 8 (pre-ReLU): max(l-real)=2.38419e-06 max(real-u)=7.62939e-06 max(u-l) slack=0
# fc4         4.76837e-07    2.86102e-06 OK
#     real logits:  [6.9972, -12.6549, -4.8748, -7.2455, -11.5148, -1.7473, -2.9593, -6.6339, -7.344, -6.7693]
#     computed low: [6.9972, -12.6549, -4.8748, -7.2455, -11.5148, -1.7473, -2.9593, -6.6339, -7.344, -6.7693]
#     computed high:[6.9972, -12.6549, -4.8748, -7.2455, -11.5148, -1.7473, -2.9593, -6.6339, -7.344, -6.7693]

# --- fc5 (img0_0.01000.txt) ---
#     layer 4 (pre-ReLU): max(l-real)=9.53674e-07 max(real-u)=7.15256e-07 max(u-l) slack=0
#     layer 6 (pre-ReLU): max(l-real)=1.43051e-06 max(real-u)=2.14577e-06 max(u-l) slack=0
#     layer 8 (pre-ReLU): max(l-real)=4.76837e-07 max(real-u)=7.15256e-07 max(u-l) slack=0
# fc5         1.43051e-06    4.76837e-07 OK
#     real logits:  [-5.4005, -4.7631, -1.702, -1.618, -3.1681, -3.5853, -12.2097, 7.0253, -5.6502, 1.4614]
#     computed low: [-5.4005, -4.7631, -1.702, -1.618, -3.1681, -3.5853, -12.2097, 7.0253, -5.6502, 1.4614]
#     computed high:[-5.4005, -4.7631, -1.702, -1.618, -3.1681, -3.5853, -12.2097, 7.0253, -5.6502, 1.4614]

# --- fc6 (img0_0.01000.txt) ---
#     layer 4 (pre-ReLU): max(l-real)=2.38419e-07 max(real-u)=4.76837e-07 max(u-l) slack=0
#     layer 6 (pre-ReLU): max(l-real)=3.57628e-07 max(real-u)=2.38419e-07 max(u-l) slack=0
#     layer 8 (pre-ReLU): max(l-real)=9.53674e-07 max(real-u)=4.76837e-07 max(u-l) slack=0
#     layer 10 (pre-ReLU): max(l-real)=1.43051e-06 max(real-u)=7.15256e-07 max(u-l) slack=0
# fc6         9.53674e-07    1.90735e-06 OK
#     real logits:  [-15.6914, -4.1364, -9.3837, 16.1794, -11.3708, -4.2336, -21.6839, -11.5284, -6.6568, -7.4484]
#     computed low: [-15.6914, -4.1364, -9.3837, 16.1794, -11.3708, -4.2336, -21.6839, -11.5284, -6.6568, -7.4484]
#     computed high:[-15.6914, -4.1364, -9.3837, 16.1794, -11.3708, -4.2336, -21.6839, -11.5284, -6.6568, -7.4484]

# --- fc7 (img0_0.50000.txt) ---
#     layer 4 (pre-ReLU): max(l-real)=1.90735e-06 max(real-u)=1.90735e-06 max(u-l) slack=0
#     layer 6 (pre-ReLU): max(l-real)=1.90735e-06 max(real-u)=1.90735e-06 max(u-l) slack=0
#     layer 8 (pre-ReLU): max(l-real)=1.19209e-07 max(real-u)=3.57628e-07 max(u-l) slack=0
#     layer 10 (pre-ReLU): max(l-real)=2.38419e-07 max(real-u)=1.86265e-07 max(u-l) slack=0
#     layer 12 (pre-ReLU): max(l-real)=3.57628e-07 max(real-u)=4.76837e-07 max(u-l) slack=0
# fc7         4.76837e-07    4.76837e-07 OK
#     real logits:  [-5.0498, -5.014, -6.5707, -4.023, -8.4514, -5.5422, -6.9269, -6.7842, -1.1206, -3.9271]
#     computed low: [-5.0498, -5.014, -6.5707, -4.023, -8.4514, -5.5422, -6.9269, -6.7842, -1.1206, -3.9271]
#     computed high:[-5.0498, -5.014, -6.5707, -4.023, -8.4514, -5.5422, -6.9269, -6.7842, -1.1206, -3.9271]

# --- conv1 (img0_0.13500.txt) ---
#     layer 3 (pre-ReLU): max(l-real)=7.7486e-07 max(real-u)=9.53674e-07 max(u-l) slack=0
#     layer 6 (pre-ReLU): max(l-real)=2.86102e-06 max(real-u)=5.72205e-06 max(u-l) slack=0
# conv1        3.8147e-06    1.90735e-06 OK
#     real logits:  [-14.3182, -11.2005, -10.3384, 15.7825, -9.2032, 1.9632, -23.3727, -16.6727, -4.333, 0.4151]
#     computed low: [-14.3182, -11.2005, -10.3383, 15.7825, -9.2032, 1.9632, -23.3727, -16.6727, -4.333, 0.4151]
#     computed high:[-14.3182, -11.2005, -10.3383, 15.7825, -9.2032, 1.9632, -23.3727, -16.6727, -4.333, 0.4151]

# --- conv2 (img0_0.13000.txt) ---
#     layer 3 (pre-ReLU): max(l-real)=1.43051e-06 max(real-u)=1.90735e-06 max(u-l) slack=0
#     layer 5 (pre-ReLU): max(l-real)=9.53674e-06 max(real-u)=1.52588e-05 max(u-l) slack=0
#     layer 8 (pre-ReLU): max(l-real)=1.78814e-06 max(real-u)=2.38419e-06 max(u-l) slack=0
# conv2       2.86102e-06    5.72205e-06 OK
#     real logits:  [-9.5953, -20.3469, -7.1597, -5.6182, -14.8903, -5.1898, -3.0497, -17.6371, 8.2638, -3.6302]
#     computed low: [-9.5953, -20.3469, -7.1597, -5.6182, -14.8903, -5.1898, -3.0497, -17.6371, 8.2638, -3.6302]
#     computed high:[-9.5953, -20.3469, -7.1597, -5.6182, -14.8903, -5.1898, -3.0497, -17.6371, 8.2638, -3.6302]

# --- conv3 (img0_0.01000.txt) ---
#     layer 3 (pre-ReLU): max(l-real)=7.15256e-07 max(real-u)=7.15256e-07 max(u-l) slack=0
#     layer 5 (pre-ReLU): max(l-real)=2.86102e-06 max(real-u)=4.76837e-06 max(u-l) slack=0
#     layer 8 (pre-ReLU): max(l-real)=1.43051e-06 max(real-u)=1.19209e-06 max(u-l) slack=0
#     layer 10 (pre-ReLU): max(l-real)=9.53674e-07 max(real-u)=1.43051e-06 max(u-l) slack=0
# conv3       1.43051e-06     3.8147e-06 OK
#     real logits:  [-9.5166, 7.1171, -6.0813, -6.8525, -2.296, -7.3818, -5.0729, -4.4684, -3.1649, -4.5996]
#     computed low: [-9.5166, 7.1171, -6.0813, -6.8525, -2.296, -7.3818, -5.0729, -4.4684, -3.1648, -4.5996]
#     computed high:[-9.5166, 7.1171, -6.0813, -6.8525, -2.296, -7.3818, -5.0729, -4.4684, -3.1648, -4.5996]

# ============================================================
# All networks match the real forward pass at eps~=0.
# This rules out a systematic bug in the core Linear/Conv2d/ReLU/