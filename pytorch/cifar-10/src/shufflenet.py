#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable


class ShuffleFunction(Function):
    @staticmethod
    def forward(ctx, input, groups):
        batch_size, channels, in_height, in_width = input.size()
        if channels % groups != 0:
            raise ValueError('invalid groups %d'.format(groups))
        shuffle_channels = torch.LongTensor(range(channels))
        channels_per_group = channels // groups
        for i in range(channels):
            shuffle_channels[i] = (i // channels_per_group) + (i % channels_per_group) * groups
        if input.is_cuda:
            shuffle_channels = shuffle_channels.cuda()
        out = input.index_select(dim=1, index=shuffle_channels)
        ctx._shuffle_channels = shuffle_channels
        return out

    @staticmethod
    def backward(ctx, grad_output):
        shuffle_channels = ctx._shuffle_channels

        reverse_shuffle_channels = torch.LongTensor(len(shuffle_channels))
        for i in range(len(shuffle_channels)):
            reverse_shuffle_channels[shuffle_channels[i]] = i

        if grad_output.is_cuda:
            reverse_shuffle_channels = reverse_shuffle_channels.cuda()

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.index_select(dim=1, index=Variable(reverse_shuffle_channels, volatile=True))
        return grad_input, None


class Shuffle(nn.Module):
    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, input):
        return ShuffleFunction.apply(input, self.groups)


def dwconv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, stride, groups):
        super(ShuffleNetUnit, self).__init__()
        self.gconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = Shuffle(groups)
        self.dwconv = dwconv3x3(in_channels, in_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.gconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.downsample = False
        if stride == 2:
            self.avg1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.downsample = True

    def forward(self, x):
        residual = x

        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.shuffle(out)
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.gconv2(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.avg1(residual)
            out = torch.cat((residual, out), dim=1)
            out = self.relu(out)
        else:
            out += residual
            out = self.relu(out)
        return out


class ShuffleNetCifar10(nn.Module):
    def __init__(self, layer_size, groups=2, input_features=16, num_classes=10):
        super(ShuffleNetCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, input_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(input_features, layer_size, groups=groups)
        self.layer2 = self.make_layer(input_features * 2, layer_size, groups=groups)
        self.layer3 = self.make_layer(input_features * 4, layer_size, groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_features * 8, num_classes)
        self.initialize(self.children())

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Sequential):
                self.initialize(list(m))
            elif isinstance(m, ShuffleNetUnit):
                self.initialize(m.children())

    def make_layer(self, in_channels, layer_size, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, stride=2, groups=groups))
        for i in range(1, layer_size):
            layers.append(ShuffleNetUnit(in_channels * 2, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def shufflenet_cifar10_1(groups=4):
    model = ShuffleNetCifar10(1, groups=groups)
    return model


def shufflenet_cifar10_3(groups=4):
    model = ShuffleNetCifar10(3, groups=groups)
    return model


def shufflenet_cifar10_4(groups=4):
    model = ShuffleNetCifar10(4, groups=groups)
    return model

def shufflenet_cifar10_6(groups=4):
    model = ShuffleNetCifar10(6, groups=groups)
    return model