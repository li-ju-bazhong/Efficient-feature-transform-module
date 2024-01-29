'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name          | layers | params
ResNet20-ftm  |    58  | 0.02M
ResNet32-ftm  |    94  | 0.03M
ResNet44-ftm  |   130  | 0.05M
ResNet56-ftm  |   166  | 0.06M
ResNet110-ftm |   328  | 0.12M
ResNet1202-ftm|  3604  | 1.39m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20_ftm', 'resnet32_ftm', 'resnet44_ftm', 
            'resnet56_ftm', 'resnet110_ftm', 'resnet164_ftm', 'resnet1202_ftm']

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NeighborFusion(nn.Module):
    def __init__(self, kernel_size=3):
        super(NeighborFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class FeatureTransModule(nn.Module):
    def __init__(self, inputC, outputC, S_ratio=2, kernel_size=1, stride=1, padding=0, bias=False):
        super(FeatureTransModule, self).__init__()
        self.out = outputC
        self.S_ratio = S_ratio
        self.k_pw2 = math.ceil(outputC/S_ratio)

        self.pw1 = nn.Conv2d(in_channels=inputC, out_channels=self.k_pw2, kernel_size=(1,1))

        self.dw = nn.Conv2d(in_channels=self.k_pw2, out_channels=self.k_pw2, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=self.k_pw2,
                                   bias=bias, padding_mode='zeros')
        
        for i in range(S_ratio - 1):
            self.add_module('NeighborFusion'+str(i), NeighborFusion(kernel_size=1))

    def forward(self, x):
        x = self.pw1(x)
        out = self.dw(x)
        for i in range(self.S_ratio - 1):
            out = torch.cat([out, getattr(self, "NeighborFusion" + str(i))(out)], dim=1)
        return out[:,:self.out,:,:]

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = FeatureTransModule(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = FeatureTransModule(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     FeatureTransModule(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = FeatureTransModule(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20_ftm(**kargs):
    return ResNet(BasicBlock, [3, 3, 3], **kargs)


def resnet32_ftm(**kargs):
    return ResNet(BasicBlock, [5, 5, 5], **kargs)


def resnet44_ftm(**kargs):
    return ResNet(BasicBlock, [7, 7, 7], **kargs)


def resnet56_ftm(**kargs):
    return ResNet(BasicBlock, [9, 9, 9], **kargs)


def resnet110_ftm(**kargs):
    return ResNet(BasicBlock, [18, 18, 18], **kargs)


def resnet164_ftm(**kargs):
    return ResNet(BasicBlock, [27, 27, 27], **kargs)


def resnet1202_ftm(**kargs):
    return ResNet(BasicBlock, [200, 200, 200], **kargs)
