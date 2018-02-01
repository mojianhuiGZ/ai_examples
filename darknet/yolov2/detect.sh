#!/bin/bash

DARKNET=/mnt/data1/projects/darknet/darknet

$DARKNET detect cfg/yolo.2.0.cfg yolo.2.0.weights $1
