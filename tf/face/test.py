# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
from six.moves import xrange
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import caltech_webfaces
from PIL import Image, ImageDraw

FLAGS = None

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1, 8))
    return images_placeholder, labels_placeholder


def inference(images, weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'decay': 0.999,
                                           'scale': True},
                        weights_initializer=slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT',
                                                                              uniform=False),
                        # weights_regularizer=slim.l2_regularizer(weight_decay)):
                        weights_regularizer=None):
        with slim.arg_scope([slim.max_pool2d], padding='SAME'):
            x = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            x = slim.conv2d(x, num_outputs=32, kernel_size=5, stride=1, padding='SAME', scope='conv1')
            x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool1')
            x = slim.conv2d(x, num_outputs=32, kernel_size=5, stride=1, padding='SAME', scope='conv2')
            x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool2')
            x = slim.conv2d(x, num_outputs=32, kernel_size=5, stride=1, padding='SAME', scope='conv3')
            x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool3')
            x = slim.conv2d(x, num_outputs=32, kernel_size=5, stride=1, padding='SAME', scope='conv4')
            x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool4')
            x = slim.flatten(x, scope='flattern')
            x = slim.fully_connected(x, 8, activation_fn=tf.nn.sigmoid, scope='fc')
    return x


def eval(sess, batch_size, images_placeholder, labels_placeholder, dataset):
    images_feed, labels_feed = dataset.get_next_batch(batch_size)
    feed_dict = {images_placeholder: images_feed, labels_placeholder: labels_feed}
    draw = ImageDraw.Draw(images_feed)
    for label in labels_feed[0]:
        draw.point((label[0], label[1]), 'red')
        draw.point((label[2], label[3]), 'red')
        draw.point((label[4], label[5]), 'red')
        draw.point((label[6], label[7]), 'red')

    tf.summary.image('image', images_feed, max_outputs=FLAGS.count)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    with tf.Graph().as_default():
        dataset = caltech_webfaces.read_dataset(FLAGS.data_dir, (IMAGE_WIDTH, IMAGE_HEIGHT))
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        result = inference(images_placeholder)

        print("------------------------------------------------------------")
        print("all model variables:")
        for w in slim.get_model_variables():
            shape = w.get_shape().as_list()
            print("  {} shape:{}".format(w.name, shape))

        print("all regularization losses:")
        for loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            print("  {}".format(loss.name))
        print("------------------------------------------------------------")

        result = tf.reshape(result, [-1, 1, 8])
        loss_mse = tf.reduce_sum(tf.square(labels_placeholder - result))
        regularization_loss = tf.losses.get_regularization_loss()
        loss = loss_mse + regularization_loss

        tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init)

        step = 0

        if FLAGS.load_last_checkpoint:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                step = sess.run(global_step)
            else:
                print('load checkpoint failed!')

        if FLAGS.only_eval:
            eval(sess, 10, images_placeholder, labels_placeholder, dataset)
        else:
            while True:
                step += 1
                step_start_time = time.time()
                images_feed, labels_feed = dataset.get_next_batch(FLAGS.batch_size)
                feed_dict = {images_placeholder: images_feed, labels_placeholder: labels_feed}
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                step_duration = time.time() - step_start_time
                print('Step %d | loss:%.4f | time %.3fs' % (step, loss_value, step_duration))

                if step % 100 == 0:
                    eval(sess, 10, images_placeholder, labels_placeholder, dataset)
                    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/',
        help='log directory'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/caltech/',
        help='data directory'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='ckpt/',
        help='checkpoints directory'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch size'
    )
    parser.add_argument(
        '--load_last_checkpoint',
        type=bool,
        default=True,
        help='load last checkpoint or not'
    )
    parser.add_argument(
        '--only_eval',
        type=bool,
        default=True,
        help='only eval'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
