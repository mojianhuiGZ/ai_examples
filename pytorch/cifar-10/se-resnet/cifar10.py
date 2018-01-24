#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from matplotlib import pyplot
from se_resnet import se_resnet20, se_resnet32
from resnet import resnet20, resnet32
from utils import Trainer
from shufflenet import shufflenet_cifar10_3, shufflenet_cifar10_4, shufflenet_cifar10_6

CIFAR10_ROOT = '../cifar-10'
BATCH_SIZE = 64
EPOCHS = 300


def main(model_name, **kwargs):
    batch_size = kwargs['batch_size']
    cuda = kwargs['cuda']

    transform_train = None
    if model_name.startswith('shufflenet'):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(CIFAR10_ROOT, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.CIFAR10(CIFAR10_ROOT, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

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

    if model_name == 'resnet20':
        model = resnet20()
    elif model_name == 'resnet32':
        model = resnet32()
    elif model_name == 'se-resnet20':
        model = se_resnet20(num_classes=10, reduction=kwargs['reduction'])
    elif model_name == 'se-resnet32':
        model = se_resnet32(num_classes=10, reduction=kwargs['reduction'])
    elif model_name == 'shufflenet3':
        model = shufflenet_cifar10_3(groups=kwargs['groups'])
    elif model_name == 'shufflenet4':
        model = shufflenet_cifar10_4(groups=kwargs['groups'])
    elif model_name == 'shufflenet6':
        model = shufflenet_cifar10_6(groups=kwargs['groups'])

    print('model name:%s' % model_name)
    print('model architecture:\n{}'.format(model))

    save_file = 'cifar10-%s.pkl' % model_name

    optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, verbose=True,
                                  threshold=1e-4, threshold_mode="abs", cooldown=3)
    trainer = Trainer(model, optimizer, F.cross_entropy, scheduler, save_file,
                      epoches_per_average_accuracy=10, cuda=cuda)

    is_load_params = input('Load CNN training parameters [Y/n]?')
    if is_load_params == 'Y' or is_load_params == 'y' or is_load_params == '':
        trainer.load()

    trainer.loop(EPOCHS, train_loader, test_loader)


if __name__ == '__main__':
    # main('resnet20', batch_size=BATCH_SIZE, cuda=True)
    # main('resnet32', batch_size=BATCH_SIZE, cuda=True)
    # main('se-resnet20', batch_size=BATCH_SIZE, reduction=16, cuda=True)
    # main('se-resnet32', batch_size=BATCH_SIZE, reduction=16, cuda=True)
    main('shufflenet3', batch_size=BATCH_SIZE, groups=4, cuda=False)
    # main('shufflenet4', batch_size=BATCH_SIZE, groups=4, cuda=True)
    # main('shufflenet6', batch_size=BATCH_SIZE, groups=4, cuda=True)
