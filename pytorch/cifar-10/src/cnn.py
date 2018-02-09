#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division

"""
Normal CNN for CIFAR10 dataset
"""

import torch.nn as nn
import math


class CNNReLU(nn.Module):
    def __init__(self, layer_size, num_classes=10, input_features=16):
        super(CNNReLU, self).__init__()

        self.conv1 = nn.Conv2d(3, input_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.build_layer(input_features, layer_size, False)
        self.layer2 = self.build_layer(input_features, layer_size, True)
        self.layer3 = self.build_layer(input_features * 2, layer_size, True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_features * 4, num_classes)
        self.initialize(self.children())

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

    def build_layer(self, in_channels, layer_size, downsample):
        blocks = []
        if downsample:
            # blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # blocks.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False))
            blocks.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(in_channels * 2))
            blocks.append(nn.ReLU(inplace=True))
            in_channels = in_channels * 2
            layer_size = layer_size - 1

        for i in range(1, layer_size + 1):
            blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(in_channels))
            blocks.append(nn.ReLU(inplace=True))

        return nn.Sequential(*blocks)

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
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))


def cnn1(**kwargs):
    model = CNNReLU(1, **kwargs)
    return model


def cnn3(**kwargs):
    model = CNNReLU(3, **kwargs)
    return model


def cnn5(**kwargs):
    model = CNNReLU(5, **kwargs)
    return model


def cnn7(**kwargs):
    model = CNNReLU(7, **kwargs)
    return model


class CNNLeakyReLU(nn.Module):
    def __init__(self, layer_size, num_classes=10, input_features=16, negative_slope=1e-2):
        super(CNNLeakyReLU, self).__init__()

        self.negative_slope = negative_slope
        self.conv1 = nn.Conv2d(3, input_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_features)
        self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.layer1 = self.build_layer(input_features, layer_size, False)
        self.layer2 = self.build_layer(input_features, layer_size, True)
        self.layer3 = self.build_layer(input_features * 2, layer_size, True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_features * 4, num_classes)
        self.initialize(self.children())

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

    def build_layer(self, in_channels, layer_size, downsample):
        blocks = []
        if downsample:
            blocks.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(in_channels * 2))
            blocks.append(nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True))
            in_channels = in_channels * 2
            layer_size = layer_size - 1

        for i in range(1, layer_size + 1):
            blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(in_channels))
            blocks.append(nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True))

        return nn.Sequential(*blocks)

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / ((1 + self.negative_slope ** 2) * n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Sequential):
                self.initialize(list(m))
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / ((1 + self.negative_slope ** 2) * n)))


def leaky_relu_cnn1(**kwargs):
    model = CNNLeakyReLU(1, **kwargs)
    return model


def leaky_relu_cnn3(**kwargs):
    model = CNNLeakyReLU(3, **kwargs)
    return model


def leaky_relu_cnn5(**kwargs):
    model = CNNLeakyReLU(5, **kwargs)
    return model


def leaky_relu_cnn7(**kwargs):
    model = CNNLeakyReLU(7, **kwargs)
    return model


class CNNPReLU(nn.Module):
    def __init__(self, layer_size, num_classes=10, input_features=16, negative_slope=0.25):
        super(CNNPReLU, self).__init__()

        self.negative_slope = negative_slope
        self.conv1 = nn.Conv2d(3, input_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_features)
        self.relu = nn.PReLU(init=negative_slope)
        self.layer1 = self.build_layer(input_features, layer_size, False)
        self.layer2 = self.build_layer(input_features, layer_size, True)
        self.layer3 = self.build_layer(input_features * 2, layer_size, True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_features * 4, num_classes)
        self.initialize(self.children())

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

    def build_layer(self, in_channels, layer_size, downsample):
        blocks = []
        if downsample:
            blocks.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(in_channels * 2))
            blocks.append(nn.PReLU(init=self.negative_slope))
            in_channels = in_channels * 2
            layer_size = layer_size - 1

        for i in range(1, layer_size + 1):
            blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(in_channels))
            blocks.append(nn.PReLU(init=self.negative_slope))

        return nn.Sequential(*blocks)

    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / ((1 + self.negative_slope ** 2) * n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Sequential):
                self.initialize(list(m))
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / ((1 + self.negative_slope ** 2) * n)))


def prelu_cnn1(**kwargs):
    model = CNNPReLU(1, **kwargs)
    return model


def prelu_cnn3(**kwargs):
    model = CNNPReLU(3, **kwargs)
    return model


def prelu_cnn5(**kwargs):
    model = CNNPReLU(5, **kwargs)
    return model


def prelu_cnn7(**kwargs):
    model = CNNPReLU(7, **kwargs)
    return model
