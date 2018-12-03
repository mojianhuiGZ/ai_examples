from __future__ import print_function

import paddle
import paddle.fluid as fluid
import sys
import numpy

try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print(
        "In the fluid 1.0, the trainer and inferencer are moving to paddle.fluid.contrib",
        file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *


BATCH_SIZE = 32
PARAMS_DIRNAME = "cifar10.inference.model"
USE_CUDA = False


def convolutional_network():
    image = fluid.layers.data(name='image', shape=[3, 32, 32], dtype='float32')
    conv1 = fluid.layers.conv2d(name='conv1', input=image, num_filters=32,
                                filter_size=5, stride=1, padding=2, act='relu',
                                param_attr=fluid.initializer.MSRAInitializer(uniform=False))
    bn1 = fluid.layers.batch_norm(name='bn1', input=conv1)
    pool1 = fluid.layers.pool2d(name='pool1', input=bn1, pool_type='max',
                                pool_size=3, pool_stride=2, pool_padding=1)
    bn2 = fluid.layers.batch_norm(name='bn2', input=pool1)
    predict = fluid.layers.fc(name='fc', input=bn2, size=10, act='softmax',
                              param_attr=fluid.initializer.MSRAInitializer(uniform=False))
    return predict


def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = convolutional_network()
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]


def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.01)


# event_handler prints training and testing info
def event_handler(event):
    if isinstance(event, EndStepEvent):
        if event.step % 10 == 0:
            print("Pass %d, Batch %d, Cost %f" % (
                event.step, event.epoch, event.metrics[0]))

    if isinstance(event, EndEpochEvent):
        avg_cost, acc = trainer.test(
            reader=test_reader, feed_order=['image', 'label'])

        print("Test with Epoch %d, avg_cost: %s, acc: %s" % (event.epoch, avg_cost, acc))

        # save parameters
        trainer.save_params(PARAMS_DIRNAME)


train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.test10(), buf_size=500),
    batch_size=BATCH_SIZE)

place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()

trainer = Trainer(
    train_func=train_program, place=place, optimizer_func=optimizer_program)

trainer.train(reader=train_reader, num_epochs=3,
              event_handler=event_handler, feed_order=['image', 'label'])

inferencer = Inferencer(
    infer_func=convolutional_network,
    param_path=PARAMS_DIRNAME, place=place)
