#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division
import tensorflow as tf
import numpy as np

# Reference 1 : http://www.tensorfly.cn/tfdoc/get_started/introduction.html
# Reference 2 : https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed

# 使用 NumPy 生成假数据, 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
