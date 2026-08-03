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
    使用可学习 ReLU lower-bound slope 的 DeepPoly 风格验证器。

    对不稳定 ReLU（l < 0 < u），使用：

        alpha * z <= ReLU(z),    alpha in [0, 1]

    其中 alpha 不再固定为 0/1，而是针对当前测试样本，用 Adam 最大化
    最差分类 margin：

        min_{j != y} (lower(logit_y) - upper(logit_j))

    每次传播对于任意 alpha in [0, 1] 都保持 sound；优化只是在 sound 的
    松弛族中寻找更紧的一组 alpha。
    """
    # 这些值可以在开发集上调整。CPU 时间紧时可把 OPT_STEPS 降到 10~15。
    OPT_STEPS = 25
    OPT_LR = 5e-2
    OPT_TEMPERATURE = 1.0
    OPT_INIT_EPS = 5e-2
    CERT_TOL = 1e-8

    model.eval()
    device = x.device
    dtype = x.dtype

    x_flat = x.detach().reshape(-1)
    input_dim = x_flat.numel()
    original_x_l = torch.clamp(x_flat - eps, 0.0, 1.0)
    original_x_u = torch.clamp(x_flat + eps, 0.0, 1.0)

    other_labels = torch.tensor(
        [label for label in range(10) if label != true_label],
        device=device,
        dtype=torch.long,
    )

    def concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u):
        """在输入 box [x_l, x_u] 上求仿射下界/上界的具体值。"""
        A_low_pos = A_low.clamp(min=0)
        A_low_neg = A_low.clamp(max=0)
        A_high_pos = A_high.clamp(min=0)
        A_high_neg = A_high.clamp(max=0)

        low = A_low_pos @ x_l + A_low_neg @ x_u + b_low
        high = A_high_pos @ x_u + A_high_neg @ x_l + b_high
        return low, high

    def propagate(alpha_params=None, initialize_alpha=False):
        """
        传播一次并返回 9 个分类 margin 下界。

        initialize_alpha=True 时，为每一个 ReLU 层建立一个与该层神经元数
        相同的可学习向量。稳定 ReLU 的 alpha 不会被使用，也不会产生梯度；
        使用整层向量可以避免优化过程中 cross mask 改变导致索引错位。
        """
        if alpha_params is None:
            alpha_params = []

        x_l = original_x_l
        x_u = original_x_u

        A_low = torch.eye(input_dim, device=device, dtype=dtype)
        A_high = torch.eye(input_dim, device=device, dtype=dtype)
        b_low = torch.zeros(input_dim, device=device, dtype=dtype)
        b_high = torch.zeros(input_dim, device=device, dtype=dtype)

        cur_shape = tuple(x.shape[-3:]) if x.dim() >= 3 else None
        relu_id = 0

        for layer in model.layers.children():
            if isinstance(layer, Normalization):
                # 标准差为正，因此区间端点顺序不变。
                mean = layer.mean.detach().to(device=device, dtype=dtype).view(-1)
                sigma = layer.sigma.detach().to(device=device, dtype=dtype).view(-1)
                x_l = (x_l - mean) / sigma
                x_u = (x_u - mean) / sigma
                continue

            if isinstance(layer, torch.nn.Linear):
                # detach 网络权重，使 autograd 只计算 alpha 的梯度。
                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                W_pos = W.clamp(min=0)
                W_neg = W.clamp(max=0)

                new_A_low = W_pos @ A_low + W_neg @ A_high
                new_A_high = W_pos @ A_high + W_neg @ A_low
                new_b_low = W_pos @ b_low + W_neg @ b_high
                new_b_high = W_pos @ b_high + W_neg @ b_low

                if b is not None:
                    new_b_low = new_b_low + b
                    new_b_high = new_b_high + b

                A_low, A_high = new_A_low, new_A_high
                b_low, b_high = new_b_low, new_b_high
                continue

            if isinstance(layer, torch.nn.Conv2d):
                if cur_shape is None:
                    raise RuntimeError('Conv2d encountered without a valid feature-map shape.')

                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                C_in, H_in, W_in = cur_shape
                W_pos = W.clamp(min=0)
                W_neg = W.clamp(max=0)

                def conv_coeff(weight, coeff):
                    # coeff: [N_prev, input_dim]
                    # 将 input_dim 当作 batch，一次卷积所有输入基向量的系数图。
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
                    expanded_bias = b.view(-1, 1, 1).expand(C_out, H_out, W_out).reshape(-1)
                    new_b_low = new_b_low + expanded_bias
                    new_b_high = new_b_high + expanded_bias

                A_low, A_high = new_A_low, new_A_high
                b_low, b_high = new_b_low, new_b_high
                cur_shape = (C_out, H_out, W_out)
                continue

            if isinstance(layer, torch.nn.Flatten):
                # 系数本来就是 flatten 后的二维矩阵，只需保留 cur_shape 供此前卷积使用。
                continue

            if isinstance(layer, torch.nn.ReLU):
                l, u = concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u)
                neuron_count = l.numel()

                neg_mask = u <= 0
                pos_mask = l >= 0
                cross_mask = (~neg_mask) & (~pos_mask)

                if initialize_alpha:
                    # 使用原 DeepPoly 0/1 heuristic 作为优化起点。
                    # 为整层建立参数，避免后续 cross_mask 改变时参数索引错位。
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

                # 即使调用方忘记投影，也保证传播中实际 slope 始终在 [0,1]。
                alpha_safe = alpha.clamp(0.0, 1.0)

                zeros = torch.zeros_like(l)
                ones = torch.ones_like(l)

                # Lower relaxation:
                #   stable negative -> 0
                #   stable positive -> identity
                #   unstable       -> alpha * z
                lower_scale = torch.where(
                    pos_mask,
                    ones,
                    torch.where(cross_mask, alpha_safe, zeros),
                )
                A_low = lower_scale.unsqueeze(1) * A_low
                b_low = lower_scale * b_low

                # Upper relaxation for unstable ReLU:
                #   ReLU(z) <= lambda * z - lambda * l,
                #   lambda = u / (u - l)
                denominator = (u - l).clamp_min(torch.finfo(dtype).eps)
                upper_slope_cross = u / denominator
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
                continue

            raise NotImplementedError('Unsupported layer type: {}'.format(type(layer)))

        low, high = concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u)
        margins = low[true_label] - high.index_select(0, other_labels)
        return margins, alpha_params

    # 第一次传播：建立 alpha，并以原来的 0/1 heuristic 初始化。
    alpha_params = []
    with torch.enable_grad():
        initial_margins, alpha_params = propagate(
            alpha_params=alpha_params,
            initialize_alpha=True,
        )

    initial_min_margin = float(initial_margins.min().detach())
    if initial_min_margin > CERT_TOL:
        return True

    # 没有 ReLU 时不存在可优化参数。
    if not alpha_params:
        return False

    optimizer = torch.optim.Adam(alpha_params, lr=OPT_LR)

    best_min_margin = initial_min_margin
    best_alpha = [alpha.detach().clone() for alpha in alpha_params]

    # 0/1 是很好的离散 heuristic，但从边界直接做投影优化容易被卡住。
    # 先保存它作为候选最优解，再把优化起点轻微移入 (0,1)。
    with torch.no_grad():
        for alpha in alpha_params:
            alpha.mul_(1.0 - 2.0 * OPT_INIT_EPS).add_(OPT_INIT_EPS)

    with torch.enable_grad():
        for _ in range(OPT_STEPS):
            optimizer.zero_grad(set_to_none=True)

            margins, _ = propagate(alpha_params=alpha_params)
            exact_min_margin = margins.min()
            current_value = float(exact_min_margin.detach())

            if current_value > best_min_margin:
                best_min_margin = current_value
                best_alpha = [alpha.detach().clone() for alpha in alpha_params]

            # 已经找到严格为正的所有分类 margin，可以立即返回。
            if current_value > CERT_TOL:
                return True

            # smooth max(-margin)，等价于平滑地最大化最差 margin；
            # 相比直接 -margins.min()，切换最差类别时梯度通常更稳定。
            loss = OPT_TEMPERATURE * torch.logsumexp(
                -margins / OPT_TEMPERATURE,
                dim=0,
            )

            if not torch.isfinite(loss):
                break

            loss.backward()
            optimizer.step()

            # Projected Adam：保证每个 slope 都位于 sound 区间 [0,1]。
            with torch.no_grad():
                for alpha in alpha_params:
                    alpha.clamp_(0.0, 1.0)

    # 检查最后一次 optimizer 更新后的 alpha（循环内部尚未 forward 这一状态）。
    with torch.no_grad():
        final_margins, _ = propagate(alpha_params=alpha_params)
        final_min_margin = float(final_margins.min())
        if final_min_margin > best_min_margin:
            best_min_margin = final_min_margin
            best_alpha = [alpha.detach().clone() for alpha in alpha_params]

        # 恢复整个优化过程中最好的 slope，而不是盲目采用最后一步。
        for alpha, saved_alpha in zip(alpha_params, best_alpha):
            alpha.copy_(saved_alpha)

        certified_margins, _ = propagate(alpha_params=alpha_params)
        return bool(certified_margins.min().item() > CERT_TOL)

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