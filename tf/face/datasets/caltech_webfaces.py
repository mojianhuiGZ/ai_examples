from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from PIL import Image, ImageDraw
from . import utils
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

SOURCE_URL = 'http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/'

IMAGES_FILE = 'Caltech_WebFaces.tar'
LABELS_FILE = 'WebFaces_GroundThruth.txt'
README_FILE = 'ReadMe.txt'

IMAGES_DIR = 'images/'


def read_dataset(data_dir, resize_to=(128, 96)):
    data_dir = os.path.normpath(data_dir)
    if not gfile.Exists(data_dir):
        gfile.MakeDirs(data_dir)

    if not gfile.Exists(os.path.join(data_dir, README_FILE)):
        utils.download(README_FILE, data_dir, SOURCE_URL)

    if not gfile.Exists(os.path.join(data_dir, LABELS_FILE)):
        utils.download(LABELS_FILE, data_dir, SOURCE_URL)

    if not gfile.Exists(os.path.join(data_dir, IMAGES_FILE)):
        utils.download(IMAGES_FILE, data_dir, SOURCE_URL)

    if not gfile.Exists(os.path.join(data_dir, IMAGES_DIR)):
        utils.untar(os.path.join(data_dir, IMAGES_FILE), extract_dir=os.path.join(data_dir, IMAGES_DIR))

    return CaltechWebFaces(data_dir, resize_to, True)


class CaltechWebFaces:
    def __init__(self, data_dir, resize_to, only_one_face):
        if not gfile.Exists(data_dir):
            raise ValueError('data directory({}) is not exists!'.format(data_dir))
        if not gfile.Exists(os.path.join(data_dir, IMAGES_DIR)):
            raise ValueError('image directory({}) is not exists!'.format(os.path.join(data_dir, IMAGES_DIR)))
        if not gfile.Exists(os.path.join(data_dir, LABELS_FILE)):
            raise ValueError('label file({}) is not exists!'.format(os.path.join(data_dir, LABELS_FILE)))

        self.data_dir = data_dir
        self.resize_to = resize_to
        self.only_one_face = only_one_face
        self.image_files = [f for f in os.listdir(os.path.join(data_dir, IMAGES_DIR)) if f.endswith('.jpg')]
        self.label_file = os.path.join(data_dir, LABELS_FILE)
        self.labels = {}
        self._read_labels()
        self._image_cache = {}
        self._label_cache = {}

    def get_next_batch(self, batch_size):
        random.shuffle(self.image_files)
        images = []
        labels = []
        i = 0
        while len(images) < batch_size:
            image, label = self._read_image_and_label(self.image_files[i])
            if self.only_one_face and len(label) > 1:
                i += 1
                continue
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
                image = np.concatenate((image, image, image), axis=2)
            image = image[np.newaxis, :]
            images.append(image)
            labels.append(label)
            i += 1
        images = np.concatenate(images, axis=0)
        return images, labels

    def _read_image_and_label(self, file):
        if file in self._image_cache:
            return self._image_cache[file], self._label_cache[file]

        path = os.path.normpath(os.path.join(os.path.join(self.data_dir, IMAGES_DIR), file))
        with open(path, 'rb') as in_file:
            img = Image.open(in_file)
            width, height = img.width, img.height
            a = b = 1
            if self.resize_to:
                img = img.resize(self.resize_to)
                a = self.resize_to[0] / width
                b = self.resize_to[1] / height

            new_labels = []
            for label in self.labels[file]:
                new_label = []
                for index, val in enumerate(label):
                    if index % 2 == 0:
                        new_label.append(val / width)
                    else:
                        new_label.append(val / height)
                new_labels.append(new_label)
            self._label_cache[file] = new_labels

            # draw = ImageDraw.Draw(img)
            # for label in new_labels:
            #     draw.point((label[0], label[1]), 'red')
            #     draw.point((label[2], label[3]), 'red')
            #     draw.point((label[4], label[5]), 'red')
            #     draw.point((label[6], label[7]), 'red')

            self._image_cache[file] = np.array(img).astype('float32')

        return self._image_cache[file], self._label_cache[file]

    def get_images_count(self):
        return len(self.image_files)

    def _read_labels(self):
        self.labels = {}
        with open(self.label_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                words = line.split()
                image_name = words[0]
                if not image_name in self.labels:
                    self.labels[image_name] = []
                self.labels[image_name].append([float(v) for v in words[1:]])
