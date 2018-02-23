#!/usr/bin/env python
# coding: utf-8

import sys
import torch
import logging
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
#from matplotlib import pyplot
from cnn import cnn1, cnn3, cnn5, cnn7
from cnn import leaky_relu_cnn1, leaky_relu_cnn3, leaky_relu_cnn5, leaky_relu_cnn7
from cnn import prelu_cnn1, prelu_cnn3, prelu_cnn5, prelu_cnn7
from mobilenet import mobile_net1, mobile_net3, mobile_net5, mobile_net7
from se_resnet import se_resnet20, se_resnet32
from resnet import resnet20, resnet32, preact_resnet20, preact_resnet32
from shufflenet import shufflenet_cifar10_1, shufflenet_cifar10_3, shufflenet_cifar10_4, shufflenet_cifar10_6
from densenet import densenet1, densenet3, densenet5, densenet7
from deformable_cnn import dcn1, dcn3, dcn5
from utils import Trainer

CIFAR10_ROOT = '../data'
BATCH_SIZE = 128
EPOCHS = 500
MIN_LR = 1e-6


def main(model_name, **kwargs):
    batch_size = kwargs['batch_size']
    cuda = kwargs['cuda']
    logfile = kwargs['logfile']

    logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=logfile,
            filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    augment_transform = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = transforms.Compose([
        augment_transform,
        normalize_transform
    ])

    transform_test = transforms.Compose([
        normalize_transform
    ])

    train_data = datasets.CIFAR10(CIFAR10_ROOT, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.CIFAR10(CIFAR10_ROOT, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    logging.info('Load CIFAR10 train data OK. data size is {}'.format(tuple(train_data.train_data.shape)))
    logging.info('Load CIFAR10 test data OK. data size is {}'.format(tuple(test_data.test_data.shape)))

    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # show train and test images

#    is_show = input('Show train images [y/N]?')
#    if is_show == 'Y' or is_show == 'y':
#        fg = pyplot.figure()
#        fg.suptitle('Train Images')
#        for i in range(32):
#            ax = pyplot.subplot(4, 8, i + 1)
#            ax.set_title('{}'.format(train_data.train_labels[i]))
#            ax.axis('off')
#            img = transforms.ToPILImage()(train_data.train_data[i])
#            img1 = augment_transform(img)
#            pyplot.imshow(img1)
#        pyplot.show()
#
#    is_show = input('Show test images [y/N]?')
#    if is_show == 'Y' or is_show == 'y':
#        fg = pyplot.figure()
#        fg.suptitle('Test Images')
#        for i in range(32):
#            ax = pyplot.subplot(4, 8, i + 1)
#            ax.set_title('{}'.format(test_data.test_labels[i]))
#            ax.axis('off')
#            pyplot.imshow(test_data.test_data[i])
#        pyplot.show()

    # training

    if model_name == 'cnn1':
        model = cnn1()
    elif model_name == 'cnn3':
        model = cnn3()
    elif model_name == 'cnn5':
        model = cnn5()
    elif model_name == 'cnn7':
        model = cnn7()
    elif model_name == 'leaky_relu_cnn1':
        model = leaky_relu_cnn1()
    elif model_name == 'leaky_relu_cnn3':
        model = leaky_relu_cnn3()
    elif model_name == 'leaky_relu_cnn5':
        model = leaky_relu_cnn5()
    elif model_name == 'leaky_relu_cnn7':
        model = leaky_relu_cnn7()
    elif model_name == 'prelu_cnn1':
        model = prelu_cnn1()
    elif model_name == 'prelu_cnn3':
        model = prelu_cnn3()
    elif model_name == 'prelu_cnn5':
        model = prelu_cnn5()
    elif model_name == 'prelu_cnn7':
        model = prelu_cnn7()
    elif model_name == 'mobile_net1':
        model = mobile_net1()
    elif model_name == 'mobile_net3':
        model = mobile_net3()
    elif model_name == 'mobile_net5':
        model = mobile_net5()
    elif model_name == 'mobile_net7':
        model = mobile_net7()
    elif model_name == 'resnet20':
        model = resnet20()
    elif model_name == 'resnet32':
        model = resnet32()
    elif model_name == 'preact_resnet20':
        model = preact_resnet20()
    elif model_name == 'preact_resnet32':
        model = preact_resnet32()
    elif model_name == 'se-resnet20':
        model = se_resnet20(num_classes=10, reduction=kwargs['reduction'])
    elif model_name == 'se-resnet32':
        model = se_resnet32(num_classes=10, reduction=kwargs['reduction'])
    elif model_name == 'shufflenet1':
        model = shufflenet_cifar10_1(groups=kwargs['groups'])
    elif model_name == 'shufflenet3':
        model = shufflenet_cifar10_3(groups=kwargs['groups'])
    elif model_name == 'shufflenet4':
        model = shufflenet_cifar10_4(groups=kwargs['groups'])
    elif model_name == 'shufflenet6':
        model = shufflenet_cifar10_6(groups=kwargs['groups'])
    elif model_name == 'densenet1':
        model = densenet1()
    elif model_name == 'densenet3':
        model = densenet3()
    elif model_name == 'densenet5':
        model = densenet5()
    elif model_name == 'densenet7':
        model = densenet7()
    elif model_name == 'dcn1':
        model = dcn1()
    elif model_name == 'dcn3':
        model = dcn3()
    elif model_name == 'dcn5':
        model = dcn5()

    logging.info('model name:%s' % model_name)
    logging.info('model architecture:\n{}'.format(model))

    save_file = 'cifar10-%s.pkl' % model_name

    optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=0, verbose=True,
                                  threshold=1e-4, threshold_mode="abs", cooldown=10)
    trainer = Trainer(model, optimizer, F.cross_entropy, scheduler, save_file,
                      epoches_per_average_accuracy=10, cuda=cuda)

#    is_load_params = input('Load CNN training parameters [Y/n]?')
#    if is_load_params == 'Y' or is_load_params == 'y' or is_load_params == '':
    trainer.load()

    trainer.loop(EPOCHS, MIN_LR, train_loader, test_loader)


if __name__ == '__main__':
    main(sys.argv[1], batch_size=BATCH_SIZE, cuda=True, reduction=16, groups=4, logfile='cifar10.log')
    # main('cnn1', batch_size=BATCH_SIZE, cuda=True)
    # main('cnn3', batch_size=BATCH_SIZE, cuda=True)
    # main('cnn5', batch_size=BATCH_SIZE, cuda=True)
    # main('cnn7', batch_size=BATCH_SIZE, cuda=True)
    # main('leaky_relu_cnn1', batch_size=BATCH_SIZE, cuda=True)
    # main('leaky_relu_cnn3', batch_size=BATCH_SIZE, cuda=True)
    # main('leaky_relu_cnn5', batch_size=BATCH_SIZE, cuda=True)
    # main('leaky_relu_cnn7', batch_size=BATCH_SIZE, cuda=True)
    # main('prelu_cnn1', batch_size=BATCH_SIZE, cuda=True)
    # main('prelu_cnn3', batch_size=BATCH_SIZE, cuda=True)
    # main('prelu_cnn5', batch_size=BATCH_SIZE, cuda=True)
    # main('prelu_cnn7', batch_size=BATCH_SIZE, cuda=True)
    # main('mobile_net1', batch_size=BATCH_SIZE, cuda=True)
    # main('mobile_net3', batch_size=BATCH_SIZE, cuda=True)
    # main('mobile_net5', batch_size=BATCH_SIZE, cuda=True)
    # main('mobile_net7', batch_size=BATCH_SIZE, cuda=True)
    # main('resnet20', batch_size=BATCH_SIZE, cuda=True)
    # main('resnet32', batch_size=BATCH_SIZE, cuda=True)
    # main('preact_resnet20', batch_size=BATCH_SIZE, cuda=True)
    # main('preact_resnet32', batch_size=BATCH_SIZE, cuda=True)
    # main('se-resnet20', batch_size=BATCH_SIZE, reduction=16, cuda=True)
    # main('se-resnet32', batch_size=BATCH_SIZE, reduction=16, cuda=True)
    # main('shufflenet1', batch_size=BATCH_SIZE, groups=4, cuda=True)
    # main('shufflenet3', batch_size=BATCH_SIZE, groups=4, cuda=True)
    # main('shufflenet4', batch_size=BATCH_SIZE, groups=4, cuda=True)
    # main('shufflenet6', batch_size=BATCH_SIZE, groups=4, cuda=True)
    # main('densenet1', batch_size=BATCH_SIZE, cuda=True)
    # main('densenet3', batch_size=BATCH_SIZE, cuda=True)
    # main('densenet5', batch_size=BATCH_SIZE, cuda=True)
    # main('densenet7', batch_size=BATCH_SIZE, cuda=True)
    # main('dcn1', batch_size=BATCH_SIZE, cuda=True)
    # main('dcn3', batch_size=BATCH_SIZE, cuda=True)
    # main('dcn5', batch_size=BATCH_SIZE, cuda=True)
