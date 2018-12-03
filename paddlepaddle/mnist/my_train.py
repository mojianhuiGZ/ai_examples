from __future__ import print_function
import paddle
import paddle.fluid as fluid
import sys
import os
import numpy as np
from PIL import Image

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
PARAMS_DIRNAME = "mnist_cnn.inference.model"
USE_CUDA = False


def single_layer_network():
    image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    predict = fluid.layers.fc(input=image, size=10, act='softmax')
    return predict


def convolutional_network():
    image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
    conv1 = fluid.layers.conv2d(name='conv1', input=image, num_filters=16,
                                filter_size=5, stride=1, padding=2, act='relu')
    bn1 = fluid.layers.batch_norm(name='bn1', input=conv1)
    pool1 = fluid.layers.pool2d(name='pool1', input=bn1, pool_type='max',
                                pool_size=3, pool_stride=2, pool_padding=1)
    bn2 = fluid.layers.batch_norm(name='bn2', input=pool1)
    predict = fluid.layers.fc(name='fc', input=bn2, size=10, act='softmax')
    return predict


def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = convolutional_network()
    # predict = single_layer_network()
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]


def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.01)


lists = []


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
        lists.append((event.epoch, avg_cost, acc))


train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.test(), buf_size=500),
    batch_size=BATCH_SIZE)

place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()

trainer = Trainer(
    train_func=train_program, place=place, optimizer_func=optimizer_program)

trainer.train(reader=train_reader, num_epochs=3,
              event_handler=event_handler, feed_order=['image', 'label'])

inferencer = Inferencer(
    infer_func=convolutional_network,
    # infer_func=single_layer_network,
    param_path=PARAMS_DIRNAME, place=place)


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im


cur_dir = cur_dir = os.getcwd()
img = load_image(cur_dir + '/image/infer_3.png')

results = inferencer.infer({'image': img})
lab = np.argsort(results)  # probs and lab are the results of one batch data
print ("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])
