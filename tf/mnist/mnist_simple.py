#!/usr/bin/env python
# coding: utf-8

# Reference 1: http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tf as tf
from tf.contrib.learn.python.learn.datasets import mnist

mnist = mnist.read_data_sets("data/", one_hot=True, source_url='http://yann.lecun.com/exdb/mnist/')

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.get_next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
