import sys
import argparse

import tf as tf
from datasets import caltech_webfaces

from capsules.core import _conv2d_wrapper, capsules_init, capsules_conv, capsules_fc

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96


def capsnet_v0(inputs, num_classes, iterations, name='CapsnetEM-V0'):
  with tf.variable_scope(name) as scope:

    nets = _conv2d_wrapper(
      inputs, shape=[5, 5, 3, 32], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv1'
    )
    nets = capsules_init(
      nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init'
    )
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1'
    )
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv2'
    )
    nets = capsules_conv(
      nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv3'
    )
    nets = capsules_fc(
      nets, num_classes, iterations=iterations, name='capsule_fc'
    )

    poses, activations = nets

  return poses, activations


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()

        dataset = caltech_webfaces.read_dataset(FLAGS.data_dir, (IMAGE_WIDTH, IMAGE_HEIGHT))

        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        poses, activations = capsnet_v0(images, num_classes=4, iterations=1, name='capsulesEM-V0')

        images, lables = dataset.get_next_batch(FLAGS.batch_size)
        images_reshape = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])


        margin_schedule_epoch_achieve_max = 10.0
        margin = tf.train.piecewise_constant(
            tf.cast(global_step, dtype=tf.int32),
            boundaries=[
                int(NUM_STEPS_PER_EPOCH * margin_schedule_epoch_achieve_max * x / 7) for x in xrange(1, 8)
            ],
            values=[
                x / 10.0 for x in range(2, 10)
            ]
        )

        loss = capsules.nets.spread_loss(
            labels, activations, margin=margin, name='spread_loss'
        )

        tf.summary.scalar(
            'losses/spread_loss', loss
        )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001
        )

        train_tensor = slim.learning.create_train_op(
            loss, optimizer, global_step=global_step, clip_gradient_norm=4.0
        )

        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_dir,
            log_every_n_steps=10,
            save_summaries_secs=60,
            saver=tf.train.Saver(max_to_keep=100),
            save_interval_secs=600,
            # yg: add session_config to limit gpu usage and allow growth
            session_config=tf.ConfigProto(
                # device_count = {
                #   'GPU': 0
                # },
                gpu_options={
                    'allow_growth': 0,
                    # 'per_process_gpu_memory_fraction': 0.01
                    'visible_device_list': '0'
                },
                allow_soft_placement=True,
                log_device_placement=False
            )
        )

        print("------------------------------------------------------------")
        print("all model variables:")
        for w in slim.get_model_variables():
            shape = w.get_shape().as_list()
            print("  {} shape:{}".format(w.name, shape))

        print("all regularization losses:")
        for loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
            print("  {}".format(loss.name))
        print("------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/',
        help='log directory'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/mnist/',
        help='data directory'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./ckpt/',
        help='directory to put checkpoints'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
