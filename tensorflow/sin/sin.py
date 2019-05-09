# coding: utf-8

# Reference 1: [用BP神经网络逼近正弦函数](https://blog.csdn.net/john_bian/article/details/79503572)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import matplotlib.pyplot as plt
import random

x = tf.placeholder(tf.float32, [1, 1])
y = tf.placeholder(tf.float32, [1, 1])
hidden_size = 8
w1 = tf.Variable(tf.random_normal([1, hidden_size]), tf.float32)
b1 = tf.Variable(tf.random_normal([1, hidden_size]), tf.float32)
w2 = tf.Variable(tf.random_normal([hidden_size, 1]), tf.float32)
b2 = tf.Variable(tf.random_normal([1]), tf.float32)
y1 = tf.sigmoid(tf.matmul(x, w1) + b1)
y2 = tf.matmul(y1, w2) + b2
e = tf.reduce_sum(tf.square(y - y2) * 100)
train_step=tf.train.GradientDescentOptimizer(1e-3).minimize(e)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
random.seed()
step = 0
for epoch in range(1000):
    print('---- epoch %d' % (epoch))
    for i in range(100):
        step += 1
        x_ = random.random() * 2 * math.pi
        y_ = math.sin(x_)
        _, y_predicted, loss = sess.run([train_step, y2, e],
                           feed_dict={x: [[x_]], y: [[y_]]})
        print('step %4d | x: %.4f y: %.4f y_predicted: %.4f loss: %.4f'
              % (i, x_, y_, y_predicted, loss))

print('---- prediction')
n = 100
x_ = list(map(lambda x: (2 * math.pi * x) / n, range(n)))
y_ = list(map(math.sin, x_))
y_predicted = list(map(lambda z: sess.run(y2, feed_dict={x: [[z]]})[0][0], x_))
for i in range(n):
    print('x: %.4f y:%.4f y_predicted:%.4f' % (x_[i], y_[i], y_predicted[i]))

plt.title('sin function simulation by neural network')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_, y_, color='red', label='sin')
plt.plot(x_, y_predicted, color='blue', label='predicted')
plt.legend()
plt.show()

sess.close()
