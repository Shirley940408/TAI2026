import torch
import torch.nn as nn


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)
        # (B, C, H, W)B: 批量大小 (Batch Size)C: 通道数 (Channels，如灰度图为 1，RGB 彩色图为 3)H: 图像高度 (Height)W: 图像宽度 (Width)
    def forward(self, x):
        return (x - self.mean) / self.sigma


class FullyConnected(nn.Module):
    '''
    这是一个全连接网络类。
    它做的事情是：
    先加一个归一化层
    再把 28*28 的图像展开成一维向量
    然后依次经过几个线性层 nn.Linear
    在每两个全连接层之间插入一个 ReLU 激活函数
    整个网络结构大致是:
    输入图像 -> 归一化 -> 展平成 784 维 -> Linear → ReLU → Linear → ReLU → ... 最终输出 10 个类别分数
    '''

    def __init__(self, device, input_size, fc_layers):
        super(FullyConnected, self).__init__()

        layers = [Normalization(device), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):
    '''
    这是一个卷积网络类。
    它做的事情是:
    先加一个归一化层
    然后依次经过几个卷积层 nn.Conv2d
    在每两个卷积层之间插入一个 ReLU 激活函数
    然后把卷积层的输出展平成一维向量
    再依次经过几个线性层 nn.Linear
    在每两个全连接层之间插入一个 ReLU 激活函数
    最终输出 10 个类别分数
'''

    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
