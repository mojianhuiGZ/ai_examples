#!/bin/bash

if [ ! -f "yolo.2.0.weights" ]; then
    wget https://pjreddie.com/media/files/yolo.2.0.weights
fi

