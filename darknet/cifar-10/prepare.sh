#!/bin/bash

if [ ! -f "data/cifar.tgz" ]; then
    if [ ! -e "data" ]; then
        mkdir data
    fi
    cd data
    wget https://pjreddie.com/media/files/cifar.tgz
    tar xzf cifar.tgz
    cd cifar
    find `pwd`/train -name \*.png > train.list
    find `pwd`/test -name \*.png > test.list
fi

if [ ! -d "backup" ]; then
    mkdir backup
fi
