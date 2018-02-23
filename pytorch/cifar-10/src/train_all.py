#!/bin/env python
# coding: utf-8

from __future__ import absolute_import, division
import os, shutil, subprocess, time

MODELS = [
'cnn1',
'cnn3',
'cnn5',
'cnn7',
'leaky_relu_cnn1',
'leaky_relu_cnn3',
'leaky_relu_cnn5',
'leaky_relu_cnn7',
'prelu_cnn1',
'prelu_cnn3',
'prelu_cnn5',
'prelu_cnn7',
'mobile_net1',
'mobile_net3',
'mobile_net5',
'mobile_net7',
'resnet20',
'resnet32',
'preact_resnet20',
'preact_resnet32',
'se-resnet20',
'se-resnet32',
'shufflenet1',
'shufflenet3',
'shufflenet4',
'shufflenet6',
'densenet1',
'densenet3',
'densenet5',
'densenet7',
'dcn1',
'dcn3',
'dcn5'
]

#CHILD_COUNT = 8
CHILD_COUNT = 1

def get_best_pkl(model):
    return 'cifar10-%s-best.pkl' % model

def get_current_pkl(model):
    return 'cifar10-%s.pkl' % model

childs = []
for m in MODELS:
    if os.path.isfile(get_best_pkl(m)):
        print('Found %s, copy to %s' % (get_best_pkl(m), get_current_pkl(m)))
        shutil.copyfile(get_best_pkl(m), get_current_pkl(m))
    while len(childs) >= CHILD_COUNT:
        time.sleep(5)
        for child in childs:
            if child['process'].poll() is None:
                continue
            else:
                print('Copy %s to %s' % (get_current_pkl(child['model']), get_best_pkl(child['model'])))
                shutil.copyfile(get_current_pkl(child['model']), get_best_pkl(child['model']))
                childs.remove(child)
    child = subprocess.Popen('python ./cifar10.py %s' % m, shell=True)
    childs.append({ 'model': m, 'process': child })

