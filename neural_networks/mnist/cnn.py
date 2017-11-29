#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as pyplot
from torch import unsqueeze
from torch import FloatTensor


MNIST_ROOT = 'mnist'
LR = 0.001
BATCH_SIZE = 50
EPOCH = 2
PARAMS_FILE = 'params.pkl'
FIGURE_FILE = 'figure.png'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),  # output shape (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (16, 14, 14)
        )

        # to reduce parameters, we can use two 3 x 3 kernel instead of 5 x 5 kernel
        # but the new CNN will learn slowly, because CNN become more deeper.

        # self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
        #     nn.Conv2d(in_channels=1,
        #               out_channels=16,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1,
        #               ),  # output shape (16, 28, 28)
        #     nn.Conv2d(in_channels=16,
        #               out_channels=16,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1,
        #               ),  # output shape (16, 28, 28)
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),  # output shape (16, 14, 14)
        # )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),  # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 7, 7)
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        return x


# prepare train and test data

train_data = MNIST(MNIST_ROOT, train=True, download=True, transform=ToTensor())
test_data = MNIST(MNIST_ROOT, train=False, download=True)

train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

print 'Load MNIST train data OK. data size is {}'.format(tuple(train_data.train_data.size()))
print 'Load MNIST test data OK. data size is {}'.format(tuple(test_data.test_data.size()))

test_X = Variable(unsqueeze(test_data.test_data, dim=1), volatile=True).type(FloatTensor) / 255.0
test_Y = test_data.test_labels


# show train and test images

is_show = raw_input('Show train images [y/N]?')
if is_show == 'Y' or is_show == 'y':
    fg = pyplot.figure()
    fg.suptitle('Train Images')
    for i in range(16):
        ax = pyplot.subplot(2, 8, i + 1)
        ax.set_title('{}'.format(train_data.train_labels[i]))
        ax.axis('off')
        pyplot.imshow(train_data.train_data[i].numpy(), cmap='gray')
    pyplot.show()

is_show = raw_input('Show test images [y/N]?')
if is_show == 'Y' or is_show == 'y':
    fg = pyplot.figure()
    fg.suptitle('Test Images')
    for i in range(16):
        ax = pyplot.subplot(2, 8, i + 1)
        ax.set_title('{}'.format(test_data.test_labels[i]))
        ax.axis('off')
        pyplot.imshow(test_data.test_data[i].numpy(), cmap='gray')
    pyplot.show()


# training

cnn = CNN()
print 'CNN architecture:\n{}'.format(cnn)

is_load_params = raw_input('Load CNN parameters [Y/n]?')
if is_load_params == 'Y' or is_load_params == 'y' or is_load_params == '':
    try:
        cnn.load_state_dict(torch.load(PARAMS_FILE))
    except IOError as e:
        pass

losses = []
is_train = raw_input('Train CNN [Y/n]?')
if is_train == 'Y' or is_train == 'y' or is_train == '':
    optimizer = Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_data_loader):
            output = cnn(Variable(x))
            loss = loss_func(output, Variable(y))
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    fg = pyplot.figure()
    fg.suptitle('loss curve')
    loss_curve, = pyplot.plot(losses, c='r')
    pyplot.grid(True)
    pyplot.savefig(FIGURE_FILE)
    is_show = raw_input('Show loss curve [y/N]?')
    if is_show == 'Y' or is_show == 'y':
        pyplot.show(fg)
    pyplot.close(fg)


# test

pred_Y = []
for i in range(len(test_X)):
    x = unsqueeze(test_X[i], dim=0)
    output = cnn(x)
    y = torch.max(output, 1)[1].data.squeeze()
    pred_Y.append(y[0])

pred_Y = torch.LongTensor(pred_Y)
accuracy = sum(pred_Y == test_Y) / float(test_Y.size(0))
print 'accuracy: %.4f' % (accuracy)

print 'Save CNN parameters to %s' % (PARAMS_FILE)
torch.save(cnn.state_dict(), PARAMS_FILE)

