import sys
import argparse

import tf as tf
from datasets import caltech_webfaces


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    with tf.Graph().as_default():
        resize_to = (128, 96)
        dataset = caltech_webfaces.read_dataset(FLAGS.data_dir, resize_to)
        images_feed, lables_feed = dataset.get_next_batch(FLAGS.count)
        images_reshape = tf.reshape(images_feed, [-1, resize_to[1], resize_to[0], 3])
        tf.summary.image('image', images_reshape, max_outputs=FLAGS.count)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            summary_all = sess.run(merged)
            summary_writer.add_summary(summary_all, 0)
            summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log_dir',
        type=str,
        default='../logs/',
        help='log directory'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data/caltech/',
        help='data directory'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=128,
        help='count of images to show'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
