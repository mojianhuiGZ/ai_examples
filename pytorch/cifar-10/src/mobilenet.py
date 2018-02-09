#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division

"""
MobileNet for CIFAR10 dataset
"""

import torch.nn as nn
import math


def dwconv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, layer_size, num_classes=10, input_features=16):
        super(MobileNet, self).__init__()

        self.conv1 = dwconv3x3(3, input_features, stride=1)
        self.layer1 = self.build_layer(input_features, layer_size, False)
        self.layer2 = self.build_layer(input_features, layer_size, True)
        self.layer3 = self.build_layer(input_features * 2, layer_size, True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_features * 4, num_classes)
        self.initialize(self.children())

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def build_layer(self, in_channels, layer_size, downsample):
        blocks = []
        if downsample:
            blocks.append(dwconv3x3(in_channels, in_channels * 2, stride=2))
            in_channels = in_channels * 2
            layer_size = layer_size - 1

        for i in range(1, layer_size + 1):
            blocks.append(dwconv3x3(in_channels, in_channels, stride=1))

        return nn.Sequential(*blocks)

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * (m.out_channels // m.groups)
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Sequential):
                self.initialize(list(m))
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))


def mobile_net1(**kwargs):
    model = MobileNet(1, **kwargs)
    return model


def mobile_net3(**kwargs):
    model = MobileNet(3, **kwargs)
    return model


def mobile_net5(**kwargs):
    model = MobileNet(5, **kwargs)
    return model


def mobile_net7(**kwargs):
    model = MobileNet(7, **kwargs)
    return model
