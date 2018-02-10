#!/bin/sh

MODELS="
cnn1
cnn3
cnn5
cnn7
leaky_relu_cnn1
leaky_relu_cnn3
leaky_relu_cnn5
leaky_relu_cnn7
prelu_cnn1
prelu_cnn3
prelu_cnn5
prelu_cnn7
mobile_net1
mobile_net3
mobile_net5
mobile_net7
resnet20
resnet32
preact_resnet20
preact_resnet32
se-resnet20
se-resnet32
shufflenet1
shufflenet3
shufflenet4
shufflenet6
densenet1
densenet3
densenet5
densenet7
dcn1
dcn3
dcn5"

for model in $MODELS
do
  best_parameter_file=cifar10-${model}-best.pkl
  parameter_file=cifar10-${model}.pkl
  if [ -f "$best_parameter_file" ]; then
    echo Found $best_parameter_file
    cp $best_parameter_file $parameter_file
  fi
  python ./cifar10.py $model
  cp $parameter_file $best_parameter_file
done

