"""
Usage:

    python case_deep_trace_worsecase.py --net fc1 --spec ../test_cases/fc1/img2_0.11500.txt --target-label 1

`--target-label` is the class index (0-9, the real MNIST label, not the
0-8 "other_labels" position) that's still failing for this case -- pull it
from your run_diagnostics.py / verifier_debug.py [dbg] output, e.g.
"target class_idx=1: margin ... certified=False" in the debug trace tells
you which position failed; this script wants the actual label index. If you
don't know which target class is failing, run with --target-label -1 first
to see the per-class margins and pick the worst one.

What it prints, in order:
  1. Per-ReLU-layer stats: how many neurons are unstable, and of those, how
     many have the box (IBP) branch winning the l=max(...)/u=min(...)
     intersection vs the affine branch winning. If box wins for most
     neurons in a layer, alpha slope optimization cannot help that layer at
     all (see earlier diagnosis).
  2. A per-layer ablation: for each ReLU layer, pretend that layer has ZERO
     relaxation error (every neuron's post-ReLU value is forced to follow
     the clean image's sign exactly, no [l,u] widening at all -- this is
     UNSOUND, for diagnosis only, never use its output as a real
     verification result), and report how much the target margin improves.
     Layers with the biggest improvement are where precision is being lost.
"""
import argparse

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']


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


def propagate(model, x, eps, true_label, target_label, zero_out_layer_idx=None):
    """
    Hybrid IBP+affine forward pass with the fixed 0/1 heuristic (no
    optimization -- deterministic, so the ablation numbers are directly
    comparable). Returns:
      margin: float, f_true - f_target lower bound
      layer_stats: list of dicts, one per ReLU layer encountered, with
        unstable_count, box_wins, affine_wins, mean_width, max_width

    If zero_out_layer_idx is not None, that ReLU layer (0-indexed among
    ReLU layers only) is forced to zero relaxation error using the clean
    image's real sign at that layer -- UNSOUND, diagnostic only.
    """
    device, dtype = x.device, x.dtype
    layers = list(model.layers.children())
    body_layers, output_layer = layers[:-1], layers[-1]

    x_detached = x.detach()
    x_flat = x_detached.reshape(-1)
    input_dim = x_flat.numel()
    x_l = torch.clamp(x_flat - eps, 0.0, 1.0)
    x_u = torch.clamp(x_flat + eps, 0.0, 1.0)

    A_low = torch.eye(input_dim, device=device, dtype=dtype)
    A_high = torch.eye(input_dim, device=device, dtype=dtype)
    b_low = torch.zeros(input_dim, device=device, dtype=dtype)
    b_high = torch.zeros(input_dim, device=device, dtype=dtype)
    box_low = x_l.reshape_as(x_detached)
    box_high = x_u.reshape_as(x_detached)

    # Track the clean image through the same layers, to get each neuron's
    # real (unperturbed) pre-activation sign for the zero_out ablation.
    real_cur = x_detached.clone()

    float_eps = torch.finfo(dtype).eps
    cur_shape = tuple(x_detached.shape[-3:]) if x_detached.dim() >= 3 else None
    layer_stats = []
    relu_idx = 0

    def concrete_bounds(A_l, b_l, A_h, b_h, xl, xu):
        A_l_pos, A_l_neg = A_l.clamp(min=0), A_l.clamp(max=0)
        A_h_pos, A_h_neg = A_h.clamp(min=0), A_h.clamp(max=0)
        low = A_l_pos @ xl + A_l_neg @ xu + b_l
        high = A_h_pos @ xu + A_h_neg @ xl + b_h
        return low, high

    with torch.no_grad():
        for layer in body_layers:
            if isinstance(layer, Normalization):
                mean = layer.mean.to(device=device, dtype=dtype).reshape(-1)
                sigma = layer.sigma.to(device=device, dtype=dtype).reshape(-1)
                x_l, x_u = (x_l - mean) / sigma, (x_u - mean) / sigma
                box_low, box_high = (box_low - layer.mean) / layer.sigma, (box_high - layer.mean) / layer.sigma
                real_cur = layer(real_cur)
                continue

            if isinstance(layer, torch.nn.Flatten):
                box_low, box_high = box_low.reshape(box_low.size(0), -1), box_high.reshape(box_high.size(0), -1)
                real_cur = layer(real_cur)
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.to(device=device, dtype=dtype)
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
                new_A_low = W_pos @ A_low + W_neg @ A_high
                new_A_high = W_pos @ A_high + W_neg @ A_low
                new_b_low = W_pos @ b_low + W_neg @ b_high
                new_b_high = W_pos @ b_high + W_neg @ b_low
                if b is not None:
                    new_b_low, new_b_high = new_b_low + b, new_b_high + b
                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high

                box_low_flat, box_high_flat = box_low.reshape(box_low.size(0), -1), box_high.reshape(box_high.size(0), -1)
                new_box_low = box_low_flat @ W_pos.t() + box_high_flat @ W_neg.t()
                new_box_high = box_high_flat @ W_pos.t() + box_low_flat @ W_neg.t()
                if b is not None:
                    new_box_low, new_box_high = new_box_low + b, new_box_high + b
                box_low, box_high = new_box_low, new_box_high
                cur_shape = None
                real_cur = layer(real_cur)
                continue

            if isinstance(layer, torch.nn.Conv2d):
                if cur_shape is None:
                    cur_shape = tuple(box_low.shape[-3:])
                C_in, H_in, W_in = cur_shape
                W = layer.weight.to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.to(device=device, dtype=dtype)
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                def conv_c(weight, A):
                    A_img = A.t().reshape(input_dim, C_in, H_in, W_in)
                    return F.conv2d(A_img, weight, None, stride=layer.stride,
                                     padding=layer.padding, dilation=layer.dilation, groups=layer.groups)

                low_img = conv_c(W_pos, A_low) + conv_c(W_neg, A_high)
                high_img = conv_c(W_pos, A_high) + conv_c(W_neg, A_low)
                C_out, H_out, W_out = low_img.shape[1:]
                new_A_low, new_A_high = low_img.reshape(input_dim, -1).t(), high_img.reshape(input_dim, -1).t()

                def conv_b(weight, bvec):
                    b_img = bvec.reshape(1, C_in, H_in, W_in)
                    return F.conv2d(b_img, weight, None, stride=layer.stride,
                                     padding=layer.padding, dilation=layer.dilation, groups=layer.groups).reshape(-1)

                new_b_low = conv_b(W_pos, b_low) + conv_b(W_neg, b_high)
                new_b_high = conv_b(W_pos, b_high) + conv_b(W_neg, b_low)
                if b is not None:
                    bias_flat = b.view(-1, 1, 1).expand(C_out, H_out, W_out).reshape(-1)
                    new_b_low, new_b_high = new_b_low + bias_flat, new_b_high + bias_flat
                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high

                new_box_low = F.conv2d(box_low, W_pos, None, stride=layer.stride, padding=layer.padding,
                                        dilation=layer.dilation, groups=layer.groups) + \
                    F.conv2d(box_high, W_neg, None, stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups)
                new_box_high = F.conv2d(box_high, W_pos, None, stride=layer.stride, padding=layer.padding,
                                         dilation=layer.dilation, groups=layer.groups) + \
                    F.conv2d(box_low, W_neg, None, stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups)
                if b is not None:
                    bias_img = b.view(1, -1, 1, 1)
                    new_box_low, new_box_high = new_box_low + bias_img, new_box_high + bias_img
                box_low, box_high = new_box_low, new_box_high
                cur_shape = (C_out, H_out, W_out)
                real_cur = layer(real_cur)
                continue

            if isinstance(layer, torch.nn.ReLU):
                affine_l, affine_u = concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u)
                box_l_flat, box_u_flat = box_low.reshape(-1), box_high.reshape(-1)
                l = torch.maximum(affine_l, box_l_flat)
                u = torch.minimum(affine_u, box_u_flat)
                u = torch.maximum(u, l)

                neg_mask, pos_mask = u <= 0, l >= 0
                cross_mask = (~neg_mask) & (~pos_mask)

                box_won = (box_l_flat > affine_l) | (box_u_flat < affine_u)
                layer_stats.append({
                    'unstable_count': int(cross_mask.sum().item()),
                    'box_wins': int((box_won & cross_mask).sum().item()) + int((box_won & neg_mask).sum().item()) + int((box_won & pos_mask).sum().item()),
                    'total_neurons': int(l.numel()),
                    'mean_width': float((u - l).mean().item()),
                    'max_width': float((u - l).max().item()),
                })

                if zero_out_layer_idx == relu_idx:
                    # UNSOUND diagnostic override: force this layer's post-ReLU
                    # value to exactly track the clean image's real sign, i.e.
                    # zero relaxation error at this layer only.
                    real_flat = real_cur.reshape(-1)
                    real_pos = real_flat >= 0
                    new_A_low = torch.where(real_pos.unsqueeze(1), A_low, torch.zeros_like(A_low))
                    new_A_high = torch.where(real_pos.unsqueeze(1), A_high, torch.zeros_like(A_high))
                    new_b_low = torch.where(real_pos, b_low, torch.zeros_like(b_low))
                    new_b_high = torch.where(real_pos, b_high, torch.zeros_like(b_high))
                    A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high
                    box_low = torch.where(real_pos.reshape_as(box_low), box_low.clamp(min=0), torch.zeros_like(box_low))
                    box_high = torch.where(real_pos.reshape_as(box_high), box_high.clamp(min=0), torch.zeros_like(box_high))
                else:
                    new_A_low, new_A_high = A_low.clone(), A_high.clone()
                    new_b_low, new_b_high = b_low.clone(), b_high.clone()
                    new_A_low[neg_mask], new_A_high[neg_mask] = 0, 0
                    new_b_low[neg_mask], new_b_high[neg_mask] = 0, 0
                    if cross_mask.any():
                        l_c, u_c = l[cross_mask], u[cross_mask]
                        slope_up = u_c / (u_c - l_c).clamp_min(float_eps)
                        new_A_high[cross_mask] = slope_up.unsqueeze(1) * A_high[cross_mask]
                        new_b_high[cross_mask] = slope_up * b_high[cross_mask] - slope_up * l_c
                        use_identity = u_c >= -l_c
                        keep_idx = cross_mask.clone()
                        keep_idx[cross_mask] = ~use_identity
                        new_A_low[keep_idx], new_b_low[keep_idx] = 0, 0
                    A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high
                    box_low, box_high = torch.clamp(l, min=0).reshape_as(box_low), torch.clamp(u, min=0).reshape_as(box_high)

                relu_idx += 1
                real_cur = layer(real_cur)
                continue

            raise NotImplementedError(f'Unsupported layer: {type(layer)}')

        W_out = output_layer.weight.to(device=device, dtype=dtype)
        b_out = output_layer.bias
        b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype) if b_out is None else b_out.to(device=device, dtype=dtype)

        if target_label is None:
            w_margin = W_out
            b_margin = b_out
        else:
            w_margin = (W_out[true_label] - W_out[target_label]).unsqueeze(0)
            b_margin = (b_out[true_label] - b_out[target_label]).unsqueeze(0)
        w_pos, w_neg = w_margin.clamp(min=0), w_margin.clamp(max=0)
        margin_A_low = w_pos @ A_low + w_neg @ A_high
        margin_b_low = w_pos @ b_low + w_neg @ b_high + b_margin
        A_pos, A_neg = margin_A_low.clamp(min=0), margin_A_low.clamp(max=0)
        affine_margin = (A_pos @ x_l + A_neg @ x_u + margin_b_low)

        box_l_flat, box_u_flat = box_low.reshape(-1), box_high.reshape(-1)
        box_margin = w_pos @ box_l_flat + w_neg @ box_u_flat + b_margin
        margin = torch.maximum(affine_margin, box_margin)

    return margin, layer_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', required=True, choices=NETWORK_NAMES)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--target-label', type=int, default=None,
                         help='Class index still failing (0-9). If omitted, automatically finds the worst '
                              '(most negative margin) class and analyzes that one. Pass -1 to only list all '
                              'per-class margins without running the full analysis.')
    args = parser.parse_args()

    true_label, pixel_values, eps = parse_spec(args.spec)
    net = load_network(args.net)
    x = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    assert net(x).max(dim=1)[1].item() == true_label

    if args.target_label == -1:
        print(f'true_label={true_label}, eps={eps}')
        print('per-class margins (f_true - f_target), negative = not yet certified:')
        for label in range(10):
            if label == true_label:
                continue
            margin, _ = propagate(net, x, eps, true_label, label)
            print(f'  class {label}: margin={margin.item():.4f}')
        return

    if args.target_label is None:
        print(f'true_label={true_label}, eps={eps}')
        print('per-class margins (f_true - f_target), negative = not yet certified:')
        worst_label, worst_margin = None, float('inf')
        for label in range(10):
            if label == true_label:
                continue
            margin, _ = propagate(net, x, eps, true_label, label)
            margin_val = margin.item()
            print(f'  class {label}: margin={margin_val:.4f}')
            if margin_val < worst_margin:
                worst_margin, worst_label = margin_val, label
        print(f'\n--target-label not given -> auto-picking the worst class: {worst_label} (margin={worst_margin:.4f})\n')
        target_label = worst_label
    else:
        target_label = args.target_label

    print(f'=== {args.net} {args.spec} | true_label={true_label} target_label={target_label} eps={eps} ===\n')

    baseline_margin, layer_stats = propagate(net, x, eps, true_label, target_label)
    print(f'baseline margin (0/1 heuristic, no optimization): {baseline_margin.item():.4f}\n')

    print('per-ReLU-layer stats:')
    for i, stats in enumerate(layer_stats):
        pct_unstable = 100.0 * stats['unstable_count'] / max(stats['total_neurons'], 1)
        print(f'  relu[{i}]: unstable={stats["unstable_count"]}/{stats["total_neurons"]} ({pct_unstable:.1f}%)  '
              f'box_wins_somewhere={stats["box_wins"]}  mean_width={stats["mean_width"]:.4f}  max_width={stats["max_width"]:.4f}')

    print('\nper-layer ablation (UNSOUND, diagnostic only -- "if this layer had zero relaxation error"):')
    results = []
    for i in range(len(layer_stats)):
        ablated_margin, _ = propagate(net, x, eps, true_label, target_label, zero_out_layer_idx=i)
        improvement = ablated_margin.item() - baseline_margin.item()
        results.append((i, ablated_margin.item(), improvement))

    results.sort(key=lambda r: -r[2])
    for i, ablated, improvement in results:
        marker = '  <-- biggest contributor' if (i, ablated, improvement) == results[0] else ''
        print(f'  zero-out relu[{i}]: margin {baseline_margin.item():.4f} -> {ablated:.4f}  (Δ={improvement:+.4f}){marker}')


if __name__ == '__main__':
    main()

# (TAIvenv310) (base) shuoyan@u172-016-131-162 code % python case_deep_trace_worsecase.py --net fc1 --spec ../test_cases/fc1/img2_0.11500.txt
# true_label=3, eps=0.115
# per-class margins (f_true - f_target), negative = not yet certified:
#   class 0: margin=-0.0278
#   class 1: margin=-3.8644
#   class 2: margin=-5.4118
#   class 4: margin=-4.6414
#   class 5: margin=-4.9821
#   class 6: margin=0.8580
#   class 7: margin=0.4140
#   class 8: margin=-4.8678
#   class 9: margin=-2.4294

# --target-label not given -> auto-picking the worst class: 2 (margin=-5.4118)

# === fc1 ../test_cases/fc1/img2_0.11500.txt | true_label=3 target_label=2 eps=0.115 ===

# baseline margin (0/1 heuristic, no optimization): -5.4118

# per-ReLU-layer stats:
#   relu[0]: unstable=35/50 (70.0%)  box_wins_somewhere=0  mean_width=10.3855  max_width=11.9998

# per-layer ablation (UNSOUND, diagnostic only -- "if this layer had zero relaxation error"):
#   zero-out relu[0]: margin -5.4118 -> 4.8905  (Δ=+10.3023)  <-- biggest contributor