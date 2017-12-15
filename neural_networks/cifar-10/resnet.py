#!/usr/bin/env python
# coding: utf-8

import time
import torch
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn
from torch.nn import functional
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot

CIFAR10_ROOT = 'cifar-10'
LR = 0.01
BATCH_SIZE = 64
EPOCH = 32
CUDA = torch.cuda.is_available()
INPUT_FEATURES = 24
RESNET_BLOCKS = [4, 6, 3]
PARAMS_FILE = 'resnet_params-{}-{}_{}_{}.pkl'.format(INPUT_FEATURES, RESNET_BLOCKS[0], RESNET_BLOCKS[1],
                                                     RESNET_BLOCKS[2])
FIGURE_FILE = 'resnet_loss-{}-{}_{}_{}.png'.format(INPUT_FEATURES, RESNET_BLOCKS[0], RESNET_BLOCKS[1],
                                                   RESNET_BLOCKS[2])


# ResNet

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = INPUT_FEATURES
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, INPUT_FEATURES, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(INPUT_FEATURES)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, INPUT_FEATURES, layers[0])
        self.layer2 = self._make_layer(block, INPUT_FEATURES * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, INPUT_FEATURES * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(INPUT_FEATURES * 4 * block.expansion, 10)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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


def predict(cnn, test_data):
    test_Y = torch.LongTensor(test_data.test_labels)
    pred_Y = []
    if CUDA:
        test_Y = test_Y.cuda()
    for i in range(len(test_data)):
        x = Variable(torch.unsqueeze(test_data[i][0], dim=0))
        if CUDA:
            x = x.cuda()
        output = cnn(x)
        y = torch.max(output, 1)[1].data.squeeze()
        pred_Y.append(y[0])
    pred_Y = torch.LongTensor(pred_Y)
    if CUDA:
        pred_Y = pred_Y.cuda()
    accuracy = sum(pred_Y == test_Y) / float(test_Y.size(0))
    return accuracy


def save_parameters(cnn):
    print('Save CNN parameters to %s' % PARAMS_FILE)
    torch.save(cnn.state_dict(), PARAMS_FILE)


# prepare train and test data

train_data = datasets.CIFAR10(CIFAR10_ROOT, train=True, download=True, transform=transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

test_data = datasets.CIFAR10(CIFAR10_ROOT, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

print('Load CIFAR10 train data OK. data size is {}'.format(tuple(train_data.train_data.shape)))
print('Load CIFAR10 test data OK. data size is {}'.format(tuple(test_data.test_data.shape)))

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# show train and test images

is_show = input('Show train images [y/N]?')
if is_show == 'Y' or is_show == 'y':
    fg = pyplot.figure()
    fg.suptitle('Train Images')
    for i in range(16):
        ax = pyplot.subplot(2, 8, i + 1)
        ax.set_title('{}'.format(train_data.train_labels[i]))
        ax.axis('off')
        pyplot.imshow(train_data.train_data[i])
    pyplot.show()

is_show = input('Show test images [y/N]?')
if is_show == 'Y' or is_show == 'y':
    fg = pyplot.figure()
    fg.suptitle('Test Images')
    for i in range(16):
        ax = pyplot.subplot(2, 8, i + 1)
        ax.set_title('{}'.format(test_data.test_labels[i]))
        ax.axis('off')
        pyplot.imshow(test_data.test_data[i])
    pyplot.show()

# training

cnn = ResNet(BasicBlock, RESNET_BLOCKS)
#cnn = ResNet(Bottleneck, RESNET_BLOCKS)
if CUDA:
    cnn = cnn.cuda()
print('CNN architecture:\n{}'.format(cnn))

is_load_params = input('Load CNN parameters [Y/n]?')
if is_load_params == 'Y' or is_load_params == 'y' or is_load_params == '':
    try:
        cnn.load_state_dict(torch.load(PARAMS_FILE))
    except IOError as e:
        pass

losses = []
is_train = input('Train CNN [Y/n]?')
if is_train == 'Y' or is_train == 'y' or is_train == '':
    optimizer = optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    if CUDA:
        loss_func = loss_func.cuda()
    for epoch in range(EPOCH):
        epoch_start_time = time.clock()
        cnn.train(True)
        for step, (x, y) in enumerate(train_data_loader):
            step_start_time = time.clock()

            if CUDA:
                x = x.cuda()
                y = y.cuda()
            xv = Variable(x)
            yv = Variable(y)
            output = cnn(xv)
            loss = loss_func(output, yv)
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_end_time = time.clock()
            print('epoch %d | loss: %.4f | using time: %.3f' % (epoch, loss.data[0], step_end_time - step_start_time))

        cnn.train(False)
        accuracy = predict(cnn, test_data)
        epoch_end_time = time.clock()
        print('epoch %d | loss: %.4f | accuracy: %.4f | using time: %.3f' % (
            epoch, loss.data[0], accuracy, epoch_end_time - epoch_start_time))
        save_parameters(cnn)

    fg = pyplot.figure()
    fg.suptitle('loss curve')
    loss_curve, = pyplot.plot(losses, c='r')
    pyplot.grid(True)
    pyplot.savefig(FIGURE_FILE)
    is_show = input('Show loss curve [y/N]?')
    if is_show == 'Y' or is_show == 'y':
        pyplot.show(fg)
    pyplot.close(fg)

else:
    cnn.train(False)
    accuracy = predict(cnn, test_data)
    print('accuracy: %.4f' % accuracy)
    save_parameters(cnn)
