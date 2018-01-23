#!/usr/bin/env python
# coding: utf-8

import time
import torch
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable, Function
from matplotlib import pyplot

CIFAR10_ROOT = '../cifar-10'
LR = 0.01
BATCH_SIZE = 64
EPOCH = 320
# CUDA = torch.cuda.is_available()
CUDA = False
INPUT_FEATURES = 16
SHUFFLENET_UNITS = [4, 4]
SHUFFLENET_GROUPS = 4
PARAMS_FILE = 'shufflenet_params-{}-{}_{}-{}.pkl'.format(INPUT_FEATURES, SHUFFLENET_UNITS[0], SHUFFLENET_UNITS[1],
                                                         SHUFFLENET_GROUPS)
FIGURE_FILE = 'shufflenet_loss-{}-{}_{}-{}.png'.format(INPUT_FEATURES, SHUFFLENET_UNITS[0], SHUFFLENET_UNITS[1],
                                                       SHUFFLENET_GROUPS)


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
        if CUDA:
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
        if CUDA:
            reverse_shuffle_channels = reverse_shuffle_channels.cuda()

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.index_select(dim=1, index=reverse_shuffle_channels)
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


class ShuffleNet(nn.Module):
    def __init__(self, in_channels, layers, groups):
        super(ShuffleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(in_channels, layers[0], groups=groups)
        self.layer2 = self.make_layer(in_channels * 2, layers[1], groups=groups)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(in_channels * 4, 10)

    def make_layer(self, in_channels, units_count, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, stride=2, groups=groups))
        for i in range(1, units_count):
            layers.append(ShuffleNetUnit(in_channels * 2, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

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

model = ShuffleNet(INPUT_FEATURES, SHUFFLENET_UNITS, groups=SHUFFLENET_GROUPS)
if CUDA:
    model = model.cuda()
print('CNN architecture:\n{}'.format(model))

is_load_params = input('Load CNN parameters [Y/n]?')
if is_load_params == 'Y' or is_load_params == 'y' or is_load_params == '':
    try:
        model.load_state_dict(torch.load(PARAMS_FILE))
    except IOError as e:
        pass

losses = []
is_train = input('Train CNN [Y/n]?')
if is_train == 'Y' or is_train == 'y' or is_train == '':

    model.train(False)
    accuracy = predict(model, test_data)
    print('accuracy: %.4f | LR: %f' % (accuracy, LR))
    max_accuracy = accuracy

    for epoch in range(EPOCH):
        epoch_start_time = time.clock()

        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss()
        if CUDA:
            loss_func = loss_func.cuda()
        model.train(True)

        for step, (x, y) in enumerate(train_data_loader):
            step_start_time = time.clock()
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            xv = Variable(x)
            yv = Variable(y)
            output = model(xv)
            loss = loss_func(output, yv)
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_end_time = time.clock()
            # print('step %d | loss is %.4f | using time: %.3f' % (step, loss.data[0], step_end_time - step_start_time))

        model.train(False)
        accuracy = predict(model, test_data)
        epoch_end_time = time.clock()
        print('epoch %d | loss: %.4f | accuracy: %.4f | LR: %f | using time: %.3f' % (
            epoch, loss.data[0], accuracy, LR, epoch_end_time - epoch_start_time))
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            save_parameters(model)
            LR = LR * 2
        elif accuracy < max_accuracy * 0.8:
            LR = LR / 2
            model.load_state_dict(torch.load(PARAMS_FILE))

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
    model.train(False)
    accuracy = predict(model, test_data)
    print('accuracy: %.4f' % accuracy)
    save_parameters(model)
