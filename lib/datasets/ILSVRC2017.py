import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
import xml.etree.ElementTree as ET

IMAGE_SIZE = 256  # Minimum image size for use with MobileNetV2
BATCH_SIZE = 16

class ILSVRC2017(object):
    def __init__(self):
        self.data_path = 'G:\\公有集数据\\ILSVRC2017Download\\2017\\ILSVRC'
        self.images_path = os.path.join(self.data_path, 'Data', 'CLS-LOC')
        self.annotations_path = os.path.join(self.data_path, 'Annotations', 'CLS-LOC')
        self.train_imagename_txt_path = os.path.join(self.data_path, 'ImageSets', 'CLS-LOC', 'train_cls.txt')
        self.val_imagename_txt_path = os.path.join(self.data_path, 'ImageSets', 'CLS-LOC', 'val.txt')
        self._classes = self._get_classes()
        self._classes_dict = dict(zip(self._classes, range(len(self._classes))))
        self._classes_index = dict(zip(range(len(self._classes)), self._classes))

    def _get_classes(self):
        cache_file = os.path.join(self.data_path, 'ImageSets', 'CLS-LOC', 'cache', 'classes_name.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fb:
                return pickle.load(fb)
        classes = os.listdir(os.path.join(self.images_path, 'train'))
        classes = sorted(classes)
        with open(cache_file, 'wb') as fb:
            pickle.dump(classes, fb)
        return classes

    def _filename_list(self, filename):
        with open(filename) as fp:
            for line in fp.readlines():
                imagename = line.split(' ')[0].split('/')[1]
                yield imagename

    def load_data_train(self):
        filenames_train_cache = os.path.join(self.data_path, 'ImageSets', 'CLS-LOC', 'cache', 'filenames_train.pkl')
        labels_train_cache = os.path.join(self.data_path, 'ImageSets', 'CLS-LOC', 'cache', 'labels_train.pkl')
        if os.path.exists(filenames_train_cache):
            with open(filenames_train_cache, 'rb') as fb:
                filenames = pickle.load(fb)
        else:
            filenames = [os.path.join(self.images_path, 'train', imagename.split('_')[0], imagename + '.jpeg')
                         for imagename in self._filename_list(self.val_imagename_txt_path)]
            with open(filenames_train_cache, 'wb') as fp:
                pickle.dump(filenames, fp)
        if os.path.exists(labels_train_cache):
            with open(labels_train_cache, 'rb') as fb:
                labels = pickle.load(fb)
        else:
            labels = [self._classes_dict[imagename.split('_')[0]]
                         for imagename in self._filename_list(self.train_imagename_txt_path)]
            with open(labels_train_cache, 'wb') as fp:
                pickle.dump(labels, fp)

        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(filenames),
                                                         tf.constant(labels)))

        train_data = (train_data.shuffle(buffer_size=1000000)
                      .map(self._parse_fn)
                      .batch(BATCH_SIZE))

        return train_data

    def _filename_val_list(self, filename):
        with open(filename) as fp:
            for line in fp.readlines():
                imagename = line.split(' ')[0]
                yield imagename

    def _get_label_from_xml(self, filename):
        filename_path = os.path.join(self.annotations_path, 'val', filename + '.xml')
        root = ET.parse(filename_path)
        objects = root.findall('object')
        return objects[0].find('name').text


    def load_data_val(self):
        filenames_val_cache = os.path.join(self.data_path, 'ImageSets', 'CLS-LOC', 'cache', 'filenames_val.pkl')
        labels_val_cache = os.path.join(self.data_path, 'ImageSets', 'CLS-LOC', 'cache', 'labels_val.pkl')
        if os.path.exists(filenames_val_cache):
            with open(filenames_val_cache, 'rb') as fb:
                filenames = pickle.load(fb)
        else:
            filenames = [os.path.join(self.images_path, 'val', imagename + '.jpeg')
                         for imagename in self._filename_val_list(self.val_imagename_txt_path)]
            with open(filenames_val_cache, 'wb') as fp:
                pickle.dump(filenames, fp)
        if os.path.exists(labels_val_cache):
            with open(labels_val_cache, 'rb') as fb:
                labels = pickle.load(fb)
        else:
            labels = [self._classes_dict[self._get_label_from_xml(imagename)]
                      for imagename in self._filename_val_list(self.val_imagename_txt_path)]
            with open(labels_val_cache, 'wb') as fp:
                pickle.dump(labels, fp)

        val_data = tf.data.Dataset.from_tensor_slices((tf.constant(filenames),
                                                         tf.constant(labels)))

        val_data = (val_data.shuffle(buffer_size=50000)
                    .map(self._parse_fn)
                    .batch(BATCH_SIZE)
                      )

        return val_data



    # Function to load and preprocess each image
    def _parse_fn(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = (tf.cast(img, tf.float32) / 127.5) - 1
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return img, label

if __name__=='__main__':
    imagenet = ILSVRC2017()
    train_data = imagenet.load_data_val()