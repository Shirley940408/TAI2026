import argparse
import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):
    """
        This is the function you are supposed to complete.
        If the network can be verified to be robust --- always predicts the true label for any perturbation within eps added to the inputs, output True; otherwise False.
    """
    with torch.no_grad():
        # Input interval: x_i - eps <= x_i <= x_i + eps
        low = inputs - eps
        high = inputs + eps

        for layer in net.layers:
            if isinstance(layer, Normalization):
                mean = layer.mean.to(low.device).to(low.dtype)
                sigma = layer.sigma.to(low.device).to(low.dtype)
                low = (low - mean) / sigma
                high = (high - mean) / sigma

            elif isinstance(layer, torch.nn.Flatten):
                low = low.reshape(low.size(0), -1)
                high = high.reshape(high.size(0), -1)

            elif isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach()
                b = layer.bias.detach()

                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)

                old_low, old_high = low, high
                low = old_low @ W_pos.t() + old_high @ W_neg.t() + b
                high = old_high @ W_pos.t() + old_low @ W_neg.t() + b

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

            elif isinstance(layer, torch.nn.ReLU):
                low = torch.clamp(low, min=0)
                high = torch.clamp(high, min=0)

            else:
                raise NotImplementedError("Unsupported layer type: %s" % type(layer))

        # After all layers: compare true label vs others
        low = low.squeeze(0)
        high = high.squeeze(0)

        true_low = low[true_label]
        other_upper = high.clone()
        other_upper[true_label] = -float('inf')

        return bool(true_low > other_upper.max())

def main():
    parser = argparse.ArgumentParser(description='Neural network verification')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        # parse the epsilon from spec file name
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

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
