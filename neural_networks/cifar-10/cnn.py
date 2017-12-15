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
PARAMS_FILE = 'cnn_params.pkl'
FIGURE_FILE = 'cnn_loss.png'


# CNN

class CNN(nn.Module):
    """ plain CNN
        parameter count = 5*5*3*24 + 24 + 24*24*5*5 + 24 + 48*24*5*5 + 48 + 48*48*5*5 + 48 + 10*3072 + 10 = 133474
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, INPUT_FEATURES, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(INPUT_FEATURES),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(INPUT_FEATURES, INPUT_FEATURES * 2, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(INPUT_FEATURES * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc1 = nn.Linear(INPUT_FEATURES * 2 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
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

cnn = CNN()
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
        start_time = time.clock()
        cnn.train(True)
        for step, (x, y) in enumerate(train_data_loader):
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

        cnn.train(False)
        accuracy = predict(cnn, test_data)
        end_time = time.clock()
        print('epoch %d | loss: %.4f | accuracy: %.4f | using time: %.3f' % (
            epoch, loss.data[0], accuracy, end_time - start_time))
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
