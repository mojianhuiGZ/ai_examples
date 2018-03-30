#!/usr/bin/env python
# coding: utf-8

# Reference 1: http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
# Reference 2: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py

"""
A MNIST classifier using CNN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import datasets


class MnistCNN:
    def __init__(self):
        self.weight_conv1 = self.weight_variable([5, 5, 1, 32], 'weight1')
        self.bias_conv1 = self.bias_variable([32], 'bias1')
        self.weight_conv2 = self.weight_variable([5, 5, 32, 64], 'weight2')
        self.bias_conv2 = self.bias_variable([64], 'bias2')
        self.weight_fc1 = self.weight_variable([7 * 7 * 64, 1024], 'weight3')
        self.bias_fc1 = self.bias_variable([1024], 'bias3')
        self.weight_fc2 = self.weight_variable([1024, 10], 'weight4')
        self.bias_fc2 = self.bias_variable([10], 'bias4')

    def forward(self, x, dropout_probablity):
        with tf.name_scope('reshape'):
            x = tf.reshape(x, [-1, 28, 28, 1])

        with tf.name_scope('conv1'):
            x = tf.nn.relu(self.conv2d(x, self.weight_conv1) + self.bias_conv1)
            x = self.max_pool_2x2(x)

        with tf.name_scope('conv2'):
            x = tf.nn.relu(self.conv2d(x, self.weight_conv2) + self.bias_conv2)
            x = self.max_pool_2x2(x)

        with tf.name_scope('fc1'):
            x = tf.reshape(x, [-1, 7 * 7 * 64])
            x = tf.nn.relu(tf.matmul(x, self.weight_fc1) + self.bias_fc1)

        with tf.name_scope('dropout'):
            x = tf.nn.dropout(x, 1 - dropout_probablity)

        with tf.name_scope('fc2'):
            x = tf.matmul(x, self.weight_fc2) + self.bias_fc2

        return x

    @staticmethod
    def weight_variable(shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    def conv2d(x, weight):
        return tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    dataset = datasets.mnist.read_data_sets("data/", source_url='http://yann.lecun.com/exdb/mnist/')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28 * 28])
        y_target = tf.placeholder(tf.int64, [None])
        dropout_probablity = tf.placeholder(tf.float32)

    y_prediction = MnistCNN().forward(x, dropout_probablity)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_target, logits=y_prediction)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_prediction, 1), y_target)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.FileWriter('logs', tf.get_default_graph()).close()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = dataset.train.next_batch(50)

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_target: batch[1], dropout_probablity: 0.})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_target: batch[1], dropout_probablity: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: dataset.test.images, y_target: dataset.test.labels, dropout_probablity: 0.}))


if __name__ == '__main__':
    main()
