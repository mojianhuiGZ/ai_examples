# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from six.moves import xrange
import time
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.learn.python.learn import datasets
from tensorflow.contrib.slim.python.slim.nets import resnet_v2


FLAGS = None

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def inference(images):
    blocks = [
        resnet_v2.resnet_v2_block('block1', base_depth=32, num_units=1, stride=2),
        resnet_v2.resnet_v2_block('block2', base_depth=64, num_units=1, stride=2),
        resnet_v2.resnet_v2_block('block3', base_depth=128, num_units=1, stride=1),
    ]

    x = tf.reshape(images, [-1, 28, 28, 1])
    net, end_points = resnet_v2.resnet_v2(x, blocks, 10, is_training=True, global_pool=True, output_stride=None,
                                include_root_block=True, reuse=None, scope='resnet_v2_50')

    # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    #     net, end_points = resnet_v2.resnet_v2_50(x, 10)

    net = tf.reshape(net, [-1, 10])
    return net


def eval(sess, eval_correct, images_placeholder, labels_placeholder, dataset):
    true_count = 0
    steps_per_epoch = dataset.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        images_feed, labels_feed = dataset.next_batch(FLAGS.batch_size, shuffle=False)
        feed_dict = {images_placeholder: images_feed, labels_placeholder: labels_feed}
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    return precision


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    with tf.Graph().as_default():
        dataset = datasets.mnist.read_data_sets(FLAGS.data_dir,
                                                source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = inference(images_placeholder)

        print("------------------------------------------------------------")
        print("all model variables:")
        for w in slim.get_model_variables():
            shape = w.get_shape().as_list()
            print("  {} shape:{}".format(w.name, shape))

        print("all regularization losses:")
        for loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            print("  {}".format(loss.name))
        print("------------------------------------------------------------")

        labels_64 = tf.to_int64(labels_placeholder)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_64, logits=logits)
        regularization_loss = tf.losses.get_regularization_loss()
        loss += regularization_loss

        tf.summary.scalar('loss', loss)

        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
        correct = tf.reduce_sum(tf.cast(correct, tf.int32))

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init)

        epoch = 0
        step = 0

        if FLAGS.load_last_checkpoint:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                step = sess.run(global_step)
                epoch = (step * FLAGS.batch_size) / dataset.train.num_examples
            else:
                print('load checkpoint failed!')

        epoch_start_time = time.time()
        while epoch < FLAGS.max_epochs:
            step += 1
            step_start_time = time.time()
            images_feed, labels_feed = dataset.train.next_batch(FLAGS.batch_size, shuffle=True)
            feed_dict = {images_placeholder: images_feed, labels_placeholder: labels_feed}
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            step_duration = time.time() - step_start_time
            print('Step %d | loss:%.4f | time %.3fs' % (step, loss_value, step_duration))

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            if (step * FLAGS.batch_size) % dataset.train.num_examples == 0:
                epoch += 1
                train_precision = eval(sess, correct, images_placeholder, labels_placeholder, dataset.train)
                validation_precision = eval(sess, correct, images_placeholder, labels_placeholder, dataset.validation)
                test_precision = eval(sess, correct, images_placeholder, labels_placeholder, dataset.test)
                epoch_duration = time.time() - epoch_start_time
                epoch_start_time = time.time()
                print('epoch:%d step:%d | precison { train:%.4f, validation:%.4f, test:%.4f } | time:%.3f '
                      % (epoch,
                         step,
                         train_precision,
                         validation_precision,
                         test_precision,
                         epoch_duration))

                checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='ckpt/',
        help='Directory to put checkpoints.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/',
        help='Directory to put the fashion mnist data.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=200,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--load_last_checkpoint',
        type=bool,
        default=True,
        help='load last checkpoint?'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    slim.learning.create_train_op()
    slim.learning.train()
