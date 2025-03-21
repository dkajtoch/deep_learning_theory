# See also: https://github.com/sgugger/Deep-Learning/blob/master/Cyclical%20LR%20and%20momentums.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor


def init_lecun_normal(m, non_linearity="relu"):
    # non_linearity="relu" is equivalent to LeCun init with a gain factor (√2).
    # non_linearity="linear" is equivalent to LeCun init with a gain factor 1.
    if isinstance(m, nn.Conv2d | nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=non_linearity)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias, 0)


def init_normal(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    else:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 0.0, 1.0)

    if hasattr(m, "bias") and m.bias is not None:
        nn.init.constant_(m.bias, 0)


class ResNetStandardBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class ResNetDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=0.1),
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


# ResNet-56 implementation in torch is based on the Caffe specification
# https://github.com/ethanhe42/resnet-cifar10-caffe/blob/master/resnet-56/trainval.prototxt
class ResNet(nn.Module):
    def __init__(self, num_blocks=(9, 9, 9), num_channels=(16, 32, 64), num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(num_features=16, momentum=0.1)
        self.block1 = self._make_block(
            16, num_channels[0], num_blocks[0], use_downsample=False
        )
        self.block2 = self._make_block(
            num_channels[0], num_channels[1], num_blocks[1], use_downsample=True
        )
        self.block3 = self._make_block(
            num_channels[1], num_channels[2], num_blocks[2], use_downsample=True
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(num_channels[2], num_classes)

    @staticmethod
    def _make_block(in_channels, out_channels, num_blocks, use_downsample=False):
        layers = []

        if use_downsample:
            layers.append(ResNetDownsampleBlock(in_channels, out_channels))
        num_remain = num_blocks - 1 if use_downsample else num_blocks
        for _ in range(num_remain):
            layers.append(ResNetStandardBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class ResNet110(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(
            num_blocks=(18, 18, 18), num_channels=(16, 32, 64), num_classes=num_classes
        )


class ResNet56(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(
            num_blocks=(9, 9, 9), num_channels=(16, 32, 64), num_classes=num_classes
        )


class ResNet44(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(
            num_blocks=(7, 7, 7), num_channels=(16, 32, 64), num_classes=num_classes
        )


class ResNet32(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(
            num_blocks=(5, 5, 5), num_channels=(16, 32, 64), num_classes=num_classes
        )


class ResNet20(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(
            num_blocks=(3, 3, 3), num_channels=(16, 32, 64), num_classes=num_classes
        )


# Based on the paper: https://arxiv.org/abs/1708.07120
class WideResNet32(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(
            num_blocks=(5, 5, 5), num_channels=(16, 64, 256), num_classes=num_classes
        )


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        stride: int = 1,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, momentum=0.1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=True
                ),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """Wide ResNet with dropout.

    Wide ResNet-28-10 is equivalent to WideResNet(28, 10, ...)

    Source: https://arxiv.org/pdf/1605.07146v2
    """

    def __init__(self, depth, widen_factor, dropout_p, num_classes):
        super().__init__()
        self.dropout_p = dropout_p

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6

        num_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(
            3, num_channels[0], kernel_size=3, stride=1, padding=1, bias=True
        )
        self.block1 = self._make_block(num_channels[0], num_channels[1], n, stride=1)
        self.block2 = self._make_block(num_channels[1], num_channels[2], n, stride=2)
        self.block3 = self._make_block(num_channels[2], num_channels[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(num_channels[3], momentum=0.9)
        self.linear = nn.Linear(num_channels[3], num_classes)

    def _make_block(self, in_channels, out_channels, num_blocks, stride):
        layers = [
            WideBasicBlock(
                in_channels, out_channels, dropout_p=self.dropout_p, stride=stride
            )
        ]
        for _ in range(1, int(num_blocks)):
            layers.append(
                WideBasicBlock(
                    out_channels, out_channels, dropout_p=self.dropout_p, stride=1
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class BnReluConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv(out)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        return out


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate):
        super().__init__()
        self.bn_relu_conv = BnReluConv(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        out = self.bn_relu_conv(x)
        out = torch.cat([x, out], 1)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.bn_relu_conv = BnReluConv(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dropout_rate=dropout_rate,
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn_relu_conv(x)
        out = self.avg_pool(out)
        return out


# Implementation follows: https://github.com/liuzhuang13/DenseNetCaffe/blob/master/make_densenet.py
class DenseNet(nn.Module):
    def __init__(
        self,
        depth=40,
        first_output=16,
        growth_rate=12,
        dropout_rate=0.2,
        num_classes=10,
    ):
        super().__init__()

        n_layers = (depth - 4) // 3

        self.conv1 = nn.Conv2d(
            3, first_output, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dense_block1 = self._make_dense_block(
            first_output, growth_rate, n_layers, dropout_rate
        )
        in_channels = first_output + n_layers * growth_rate

        self.transition1 = TransitionLayer(in_channels, in_channels, dropout_rate)
        self.dense_block2 = self._make_dense_block(
            in_channels, growth_rate, n_layers, dropout_rate
        )
        in_channels = in_channels + n_layers * growth_rate

        self.transition2 = TransitionLayer(in_channels, in_channels, dropout_rate)
        self.dense_block3 = self._make_dense_block(
            in_channels, growth_rate, n_layers, dropout_rate
        )
        in_channels = in_channels + n_layers * growth_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_dense_block(in_channels, growth_rate, n_layers, dropout_rate):
        layers = []
        for i in range(n_layers):
            layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, dropout_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.dense_block1(out)
        out = self.transition1(out)

        out = self.dense_block2(out)
        out = self.transition2(out)

        out = self.dense_block3(out)

        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class NTKLinear(nn.Linear):
    """Linear layer with NTK normalization.

    See Also:
        https://proceedings.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        beta: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_buffer("ntk_scaling", torch.tensor(1.0 / in_features).sqrt())
        self.register_buffer("beta", torch.tensor(beta))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight * self.ntk_scaling, self.bias * self.beta)


class FullyConnectedNet(nn.Module):
    """Fully connected neural network."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: list[int],
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
        use_ntk_normalization: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.act_fn = nn.ReLU()
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_ntk_normalization = use_ntk_normalization

        self.fc_layers = nn.ModuleList(
            [
                self.build_fc_block(
                    self.input_dim if i == 0 else self.hidden_sizes[i - 1],
                    self.hidden_sizes[i],
                )
                for i in range(len(self.hidden_sizes))
            ]
        )
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.num_classes)

    def build_fc_block(self, in_features, out_features):
        layers = []
        if self.use_ntk_normalization:
            layers.append(NTKLinear(in_features, out_features))
        else:
            layers.append(nn.Linear(in_features, out_features))

        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(self.act_fn)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
