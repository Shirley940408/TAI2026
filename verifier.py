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
    DeepPoly 风格的鲁棒性验证器（线性/仿射界传播版本)。

    核心区别（相对于纯区间传播 low/high):
    对网络中的每一个神经元 z,我们不再只维护两个数字 [low, high],
    而是维护两条关于"原始输入像素"x 的仿射（线性)表达式:

        A_low @ x + b_low  <=  z  <=  A_high @ x + b_high      对所有 x in [x_l, x_u]

    这样在每一层做传播时,Linear / Conv2d 的线性组合是"精确"的
    （仿射函数的正/负系数分解组合,不会引入额外误差);
    只有在 ReLU 处,才需要引入近似（这也是 DeepPoly 唯一近似的地方)。
    最后再用输入区间对最终的仿射表达式做一次区间求值,得到具体数值下界/上界。

    注意:这里选择把仿射系数直接维护到"原始输入"（而不是像标准 DeepPoly
    论文那样只维护到"上一层",再在需要时反向代入 back-substitution)。
    这样实现更直接、更不容易出错,数学上同样是 sound 的,
    只是理论上界可能比完整 back-substitution 版本略松（对 MNIST 这种规模的网络影响很小)。
    """
    device = x.device
    x_flat = x.reshape(-1)
    D = x_flat.numel()
    x_l = torch.clamp(x_flat - eps, 0.0, 1.0)
    x_u = torch.clamp(x_flat + eps, 0.0, 1.0)

    # 初始状态:输入自己相对自己的仿射表达式就是恒等映射 A=I, b=0
    A_low = torch.eye(D, device=device)
    A_high = torch.eye(D, device=device)
    b_low = torch.zeros(D, device=device)
    b_high = torch.zeros(D, device=device)

    # 记录当前特征图的空间形状（供 Conv2d 使用),x 的原始 shape 例如 (1,28,28)
    cur_shape = tuple(x.shape[-3:]) if x.dim() >= 3 else None

    def concrete_bounds(A_low, b_low, A_high, b_high):
        """把仿射表达式对输入区间 [x_l, x_u] 做区间求值,得到具体数值 low/high"""
        Al_pos, Al_neg = A_low.clamp(min=0), A_low.clamp(max=0)
        Au_pos, Au_neg = A_high.clamp(min=0), A_high.clamp(max=0)
        low = Al_pos @ x_l + Al_neg @ x_u + b_low
        high = Au_pos @ x_u + Au_neg @ x_l + b_high
        return low, high

    for layer in model.layers.children():
        if isinstance(layer, Normalization):
            mean = layer.mean.view(-1)
            sigma = layer.sigma.view(-1)

            x_l = (x_l - mean) / sigma
            x_u = (x_u - mean) / sigma
            continue
        # ---------------- Linear 层:精确的仿射组合 ----------------
        if isinstance(layer, torch.nn.Linear):
            W, b = layer.weight, layer.bias
            W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

            new_A_low = W_pos @ A_low + W_neg @ A_high
            new_A_high = W_pos @ A_high + W_neg @ A_low
            new_b_low = W_pos @ b_low + W_neg @ b_high
            new_b_high = W_pos @ b_high + W_neg @ b_low
            if b is not None:
                new_b_low = new_b_low + b
                new_b_high = new_b_high + b

            A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high

        # ---------------- Conv2d 层:把系数矩阵当成"批量图像"做卷积 ----------------
        elif isinstance(layer, torch.nn.Conv2d):
            W, b = layer.weight, layer.bias
            C_in, H_in, W_in = cur_shape
            W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

            def conv_c(Wt, A):
                # A: [N_prev, D] -> 把 D 维当作 batch,每个通道图像做卷积
                A_img = A.t().reshape(D, C_in, H_in, W_in)
                return F.conv2d(
                    A_img, Wt, None,
                    stride=layer.stride, padding=layer.padding,
                    dilation=layer.dilation, groups=layer.groups,
                )  # [D, C_out, H_out, W_out]

            low_img = conv_c(W_pos, A_low) + conv_c(W_neg, A_high)
            high_img = conv_c(W_pos, A_high) + conv_c(W_neg, A_low)
            C_out, H_out, W_out = low_img.shape[1:]
            new_A_low = low_img.reshape(D, -1).t()
            new_A_high = high_img.reshape(D, -1).t()

            # 常数项 b_low/b_high 本身也是一张"图像"（每个像素一个常数),同样做卷积
            def conv_b(Wt, bvec):
                b_img = bvec.reshape(1, C_in, H_in, W_in)
                return F.conv2d(
                    b_img, Wt, None,
                    stride=layer.stride, padding=layer.padding,
                    dilation=layer.dilation, groups=layer.groups,
                ).reshape(-1)

            new_b_low = conv_b(W_pos, b_low) + conv_b(W_neg, b_high)
            new_b_high = conv_b(W_pos, b_high) + conv_b(W_neg, b_low)
            if b is not None:
                bias_flat = b.view(-1, 1, 1).expand(C_out, H_out, W_out).reshape(-1)
                new_b_low = new_b_low + bias_flat
                new_b_high = new_b_high + bias_flat

            A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high
            cur_shape = (C_out, H_out, W_out)

        elif isinstance(layer, torch.nn.Flatten):
            continue  # 不改变仿射表达式,只是形状记账（下一层若是 Linear 无需额外处理)

        # ---------------- ReLU 层:DeepPoly 的核心近似发生在这里 ----------------
        elif isinstance(layer, torch.nn.ReLU):
            l, u = concrete_bounds(A_low, b_low, A_high, b_high)

            new_A_low = A_low.clone()
            new_A_high = A_high.clone()
            new_b_low = b_low.clone()
            new_b_high = b_high.clone()

            # 情况一:恒为负 (u <= 0) -> 输出恒为 0
            neg_mask = u <= 0
            new_A_low[neg_mask] = 0
            new_A_high[neg_mask] = 0
            new_b_low[neg_mask] = 0
            new_b_high[neg_mask] = 0

            # 情况二:恒为正 (l >= 0) -> ReLU 是恒等映射,系数不变（已经 clone 保留)

            # 情况三:跨越 0 (l < 0 < u) -> 需要线性松弛
            cross_mask = (l < 0) & (u > 0)
            if cross_mask.any():
                idx = cross_mask.nonzero(as_tuple=True)[0]
                l_c, u_c = l[idx], u[idx]

                # 上界:唯一穿过 (l,0) 和 (u,u) 两点的直线,斜率 u/(u-l)
                slope = u_c / (u_c - l_c)
                new_A_high[idx] = slope.unsqueeze(1) * A_high[idx]
                new_b_high[idx] = slope * b_high[idx] - slope * l_c

                # 下界:在 y=0 和 y=z 两条候选直线中选面积更小的一条
                # （标准 DeepPoly 的启发式;如果想要更紧的界,可以把这个 0/1
                #  换成一个可学习的 slope in [0,1],用梯度上升去优化,
                #  这就是课程提到的 "optimized ReLU relaxation slope")
                use_identity = u_c >= -l_c
                zero_idx = idx[~use_identity]
                new_A_low[zero_idx] = 0
                new_b_low[zero_idx] = 0
                # use_identity 为 True 的部分保持 slope=1,即沿用原来的 A_low/b_low

            A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high

        else:
            raise NotImplementedError("Unsupported layer type: %s" % type(layer))

    # 最终输出层:对仿射表达式做一次区间求值,得到具体的 logits 下界/上界
    low, high = concrete_bounds(A_low, b_low, A_high, b_high)

    true_low = low[true_label]
    other_upper = high.clone()
    other_upper[true_label] = -float("inf")
    return bool(true_low > other_upper.max())

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