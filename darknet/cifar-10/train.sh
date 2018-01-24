#!/bin/bash

if [ -f "backup/cifar_small.backup" ]; then
    echo -n "Load CNN parameters [Y/n]? "
    read is_load_params
    is_load_params=`echo $is_load_params | tr '[a-z]' '[A-Z]'`
    if [ "$is_load_params" == "Y" ]; then
        ./darknet classifier train cfg/cifar.data cfg/cifar_small.cfg backup/cifar_small.backup
        exit 0
    fi
fi

/opt/darknet/darknet classifier train cfg/cifar.data cfg/cifar_small.cfg

