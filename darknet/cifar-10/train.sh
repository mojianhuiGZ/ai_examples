#!/bin/bash

#DARKNET=/opt/darknet/darknet
DARKNET=/mnt/data1/projects/darknet/darknet
#DARKNET=/cygdrive/d/my/darknet/darknet

if [ -f "backup/cifar_small.backup" ]; then
    echo -n "Load CNN parameters [Y/n]? "
    read is_load_params
    is_load_params=`echo $is_load_params | tr '[a-z]' '[A-Z]'`
    if [ "$is_load_params" == "Y" ]; then
        $DARKNET classifier train cfg/cifar.data cfg/cifar_small.cfg backup/cifar_small.backup
        exit 0
    fi
fi

$DARKNET classifier train cfg/cifar.data cfg/cifar_small.cfg

