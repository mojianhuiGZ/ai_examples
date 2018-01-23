#!/usr/bin/env python
# coding: utf-8

import time
import torch
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot
from collections import OrderedDict

CIFAR10_ROOT = '../cifar-10'
MAX_LR = 0.02
LR = 0.02
BATCH_SIZE = 64
EPOCH = 320
CUDA = torch.cuda.is_available()
INPUT_FEATURES = 16
DENSENET_LAYERS = [6, 6, 6]
PARAMS_FILE = 'densenet_params-{}-{}_{}_{}.pkl'.format(INPUT_FEATURES, DENSENET_LAYERS[0], DENSENET_LAYERS[1],
                                                       DENSENET_LAYERS[2])
FIGURE_FILE = 'densenet_loss-{}-{}_{}_{}.png'.format(INPUT_FEATURES, DENSENET_LAYERS[0], DENSENET_LAYERS[1],
                                                     DENSENET_LAYERS[2])


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=8,
                 num_classes=10):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)

        return out


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

model = DenseNet(num_init_features=INPUT_FEATURES, growth_rate=INPUT_FEATURES // 2,
                 block_config=DENSENET_LAYERS, num_classes=10, drop_rate=0)
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
            if LR > MAX_LR:
                LR = MAX_LR
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
