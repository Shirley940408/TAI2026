import argparse
import os

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']


def analyze(net, inputs, eps, true_label):
    """
        This is the function you are supposed to complete.
        If the network can be verified to be robust --- always predicts the true label for any perturbation within eps added to the inputs, output True; otherwise False.
    """
    with torch.no_grad():
        '''
        输入图像原本是一个确定点；
        现在假设它在 xi - ϵ 到 xi + ϵ 这个范围内变化；
​        所以我们用两个张量来表示：
        - low: 所有像素的下界
        - high: 所有像素的上界
        也就是说,代码不是在算“一个具体输入”,而是在算“所有可能输入的范围”。

        然后后面循环里对每一层都做一次“区间传播”, 对于每一层的输出, 我们也得到一个区间(low, high)。
        最终输出的 low 和 high 就是所有可能输出的范围。
        如果最终输出的 low[true_label] > high[other_labels].max(), 那么就说明对于所有可能的输入, 网络都预测 true_label, 也就是网络是robust的, 返回 True; 否则返回 False。

        '''
        low = inputs - eps
        high = inputs + eps

        for layer in net.layers:
            # 对归一化层：直接把上下界都做归一化
            if isinstance(layer, Normalization):
                mean = layer.mean.to(low.device).to(low.dtype)
                sigma = layer.sigma.to(low.device).to(low.dtype)
                low = (low - mean) / sigma
                high = (high - mean) / sigma

            # 对展平层：把 2D 的特征图展开成向量
            elif isinstance(layer, torch.nn.Flatten):
                low = low.reshape(low.size(0), -1)
                high = high.reshape(high.size(0), -1)

            # 对线性层：用正权重和负权重分开处理, 这样就能得到新的下界和上界
            elif isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach()
                b = layer.bias.detach()

                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)

                old_low, old_high = low, high
                low = old_low @ W_pos.t() + old_high @ W_neg.t() + b
                high = old_high @ W_pos.t() + old_low @ W_neg.t() + b

            # 对卷积层：用正权重和负权重分开做卷积, 这样就能得到新的下界和上界
            elif isinstance(layer, torch.nn.Conv2d):
                W = layer.weight.detach()
                b = layer.bias.detach() if layer.bias is not None else None

                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)

                old_low, old_high = low, high

                low_conv = F.conv2d(
                    old_low, W_pos, None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups
                )
                low_conv += F.conv2d(
                    old_high, W_neg, None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups
                )

                high_conv = F.conv2d(
                    old_high, W_pos, None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups
                )
                high_conv += F.conv2d(
                    old_low, W_neg, None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups
                )

                if b is not None:
                    low_conv = low_conv + b.view(1, -1, 1, 1)
                    high_conv = high_conv + b.view(1, -1, 1, 1)

                low, high = low_conv, high_conv

            # 对 ReLU 层：把负数都变成 0, 也就是把下界和上界都按 ReLU 的效果做近似
            elif isinstance(layer, torch.nn.ReLU):
                low = torch.clamp(low, min=0)
                high = torch.clamp(high, min=0)

            else:
                raise NotImplementedError("Unsupported layer type: %s" % type(layer))

        low = low.squeeze(0)
        high = high.squeeze(0)

        '''
        先看真实类别的下界 true_low
        再看所有其他类别的上界 other_upper
        如果真实类别的最小可能输出始终大于其他类别的最大可能输出,就说明在这个扰动范围内,模型一定还是选真实类别
        lower(y_true) > max_(j!=true) upper(y_j)
        如果这个成立就说明可以验证通过
        '''
        true_low = low[true_label]
        other_upper = high.clone()
        other_upper[true_label] = -float('inf')

        return bool(true_low > other_upper.max())
    '''
严格来说,当前这个实现“不是完整意义上的 DeepPoly”,更像是“DeepPoly 思想的一个很粗糙的初级版本”。

原因有三个：

1. 它只用了最简单的区间传播
你现在做的是：
- 输入用一个上下界区间表示；
- 每层都把这个区间往后传播；
- 最后看是否能保证真实类别始终领先。

这已经有“抽象解释 / convex relaxation”的味道了,但是它没有真正做出 DeepPoly 那种“更精细的线性上界和下界近似”。

2. ReLU 的处理太粗
DeepPoly 的关键之一是对 ReLU 的处理要更精细。你现在只是：

```python
low = torch.clamp(low, min=0)
high = torch.clamp(high, min=0)
```

这相当于把所有负值都直接截成 0,忽略了很多“激活前后可能的关系”。这会让区间变得很松,结果很容易判成不稳定。

3. 你没有真正维护“每个神经元的线性上界/下界”
DeepPoly 的核心不是简单地维护一个区间,而是维护每个神经元的一个更有约束的近似关系,例如：
- 下界是一个线性表达式
- 上界是一个线性表达式

而你现在只是维护：
- 一个 `low`
- 一个 `high`

这虽然能跑通,但它是“更保守、更容易失效”的版本。

所以比较准确的说法是：

- 你现在的实现：是“基于区间的松弛传播”
- 不是“完整的 DeepPoly 实现”

如果你要把它改成更接近 DeepPoly 的版本,通常要做的不是“换一个判定公式”,而是要把每层神经元的上下界从“单纯的数值区间”升级为“更细的线性约束”。

可以把它这样理解：
- 你现在这版：像是在用一个很大的盒子包住所有可能输出；
- DeepPoly: 想用一个更贴近真实情况的“斜着的盒子/线性边界”去包住它。

所以结论就是：
- 你现在的版本“借鉴了 DeepPoly 的思路”,但“还没有真正实现 DeepPoly”。
    '''


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