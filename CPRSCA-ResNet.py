import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as t

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class CoordinateAttention(nn.Module):
    def __init__(self, dim: int, groups: int = 16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, dim // groups)
        self.conv1 = DepthwiseSeparableConv2d(dim, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = DepthwiseSeparableConv2d(mip, dim, kernel_size=1, stride=1, padding=0)
        self.conv3 = DepthwiseSeparableConv2d(mip, dim, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(mip)
        self.relu = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()

        # 水平和垂直方向池化
        x_h = self.pool_h(x)                 # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, 1, W]

        # 拼接处理
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        # 拆分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 注意力权重
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        # 扩展到原图大小
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        return x * x_w * x_h

class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)       # 全局平均池化
        y = self.fc(y).view(b, c, 1, 1)       # 全连接层生成权重
        return x * y.expand_as(x)             # 通道加权后输出

class CPRMBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 3,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            gate_layer: str = 'sigmoid',
            groups: int = 8,
    ):
        super(CPRMBlock, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        # 分组卷积部分
        self.local_dwc = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分组卷积 + 拼接
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x, self.group_chans, dim=1)
        m_s_x = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        return m_s_x

class CPRSCABlock(nn.Module):
    expansion = 1  # 添加这一行
    def __init__(self, in_channels, out_channels, stride=1):
        super(CPRSCABlock, self).__init__()
        mid_channels = out_channels //2
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.mish = Mish()
        self.cprm = CPRMBlock(mid_channels, 8)
        self.se = SqueezeExcitation(mid_channels, 8)
        self.ca = CoordinateAttention(mid_channels, 8)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x,):
        out = self.mish(self.bn1(self.conv1x1(x)))
        out = self.cprm(out)
        out1 = self.ca(out)
        out2 = self.se(out)

        out = torch.cat([out2, out1], dim=1)

        out += self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, in_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 128
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=(1, 5), stride=(1, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.dropout = nn.Dropout(0.5)
        self.layer1 = self._make_layer(block, 128, layers[0])  # out Channel
        self.layer2 = self._make_layer(block, 264, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1) # 添加一维
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(in_channels=1, num_classes=2):
    return ResNet(CPRSCABlock, [2, 2, 2, 2], num_classes, in_channels)

def ResNet34(in_channels=1, num_classes=2):
    return ResNet(CPRSCABlock, [3, 4, 6, 3], num_classes, in_channels)

def ResNet50(in_channels=1, num_classes=2):
    return ResNet(CPRSCABlock, [3, 4, 6, 3], num_classes, in_channels)

def ResNet101(in_channels=1, num_classes=2):
    return ResNet(CPRSCABlock, [3, 4, 23, 3], num_classes, in_channels)

def ResNet152(in_channels=1, num_classes=2):
    return ResNet(CPRSCABlock, [3, 8, 36, 3], num_classes, in_channels)


if __name__ == '__main__':
    # 测试网络结构
    input_data = torch.randn(8, 18, 1280)
    model = ResNet34(in_channels=1, num_classes=2)
    output = model(input_data)
    print(output.shape)  # 输出预测结果的形状
