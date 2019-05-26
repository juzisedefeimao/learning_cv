import tensorflow as tf
import numpy as np
import pandas as pd
import os
from os import path as ops
import glog as log
import json
import tqdm
import time
from typing import Iterable
import pickle

IMAGE_WIDTH = 280
IMAGE_HIGTH = 32
BATCH_SIZE = 32

class ChineseString():
    def __init__(self):
        """
        init crnn data producer
        :param dataset_dir: image dataset root dir
        :param char_dict_path: char dict path
        :param ord_map_dict_path: ord map dict path
        """
        dataset_dir = 'G:\\公有集数据\\chinese string'
        char_dict_path = 'D:\\jjj\\GitHub\\cv_learning\\lib\\datasets\\char_dict\\char_dict_cn.json'
        ord_map_dict_path = 'D:\\jjj\\GitHub\\cv_learning\\lib\\datasets\\char_dict\\ord_map_cn.json'
        if not ops.exists(dataset_dir):
            raise ValueError('Dataset dir {:s} not exist'.format(dataset_dir))

        # Check image source data
        self._dataset_dir = dataset_dir
        self._train_annotation_file_path = ops.join(dataset_dir, 'label', 'train.txt')
        self._test_annotation_file_path = ops.join(dataset_dir, 'label', 'test.txt')
        self._lexicon_file_path = ops.join(dataset_dir, 'lexicon.txt')
        self._char_dict_path = char_dict_path
        self._ord_map_dict_path = ord_map_dict_path

        # if not self._is_source_data_complete():
        #     raise ValueError('Source image data is not complete, '
        #                      'please check if one of the image folder '
        #                      'or index file is not exist')

        # Init training example information
        self._lexicon_list = []
        self._train_sample_infos = []
        self._test_sample_infos = []
        self._init_dataset_sample_info()

        # Check if need generate char dict map
        if char_dict_path is None or ord_map_dict_path is None:
            os.makedirs('./data/char_dict', exist_ok=True)
            self._char_dict_path = ops.join('./data/char_dict', 'char_dict.json')
            self._ord_map_dict_path = ops.join('./data/char_dict', 'ord_map.json')
            self._generate_char_dict()

    def load_data(self):

        # 分割训练数据为训练集和验证集
        train_size = 2752000
        _train_sample_infos_path = self._train_sample_infos[0][:train_size]
        _train_sample_infos_label = self._train_sample_infos[1][:train_size]
        _val_sample_infos_path = self._train_sample_infos[0][train_size:3279520]
        _val_sample_infos_label = self._train_sample_infos[1][train_size:3279520]


        # 加载训练数据
        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(_train_sample_infos_path),
                                                         tf.constant(_train_sample_infos_label)))

        train_data = train_data.shuffle(buffer_size=10000).map(self._parse_fn).batch(BATCH_SIZE)

        # 加载验证数据
        val_data = tf.data.Dataset.from_tensor_slices((tf.constant(_val_sample_infos_path),
                                                        tf.constant(_val_sample_infos_label)))

        val_data = val_data.map(self._parse_fn).batch(BATCH_SIZE)


        # 加载测试数据
        size = len(self._test_sample_infos[0]) - len(self._test_sample_infos[0])%BATCH_SIZE
        _test_sample_infos_path = self._test_sample_infos[0][:size]
        _test_sample_infos_label = self._test_sample_infos[1][:size]



        test_data = tf.data.Dataset.from_tensor_slices((tf.constant(_test_sample_infos_path),
                                                         tf.constant(_test_sample_infos_label)))

        test_data = test_data.map(self._parse_fn).batch(BATCH_SIZE)

        return train_data, val_data, test_data

    def load_data_sparse(self):

        # 分割训练数据为训练集和验证集
        train_size = 2752000
        _train_sample_infos_path = self._train_sample_infos[0][:train_size]
        _train_sample_infos_label = self._train_sample_infos[1][:train_size]
        _val_sample_infos_path = self._train_sample_infos[0][train_size:3279520]
        _val_sample_infos_label = self._train_sample_infos[1][train_size:3279520]

        # 测试集与验证集每一epoch需要迭代的次数
        self.train_steps_per_epoch = len(_train_sample_infos_path) / BATCH_SIZE
        self.val_steps_per_epoch = len(_val_sample_infos_path) / BATCH_SIZE

        # 加载训练数据
        train_label_sparse = self._label_to_sparse(_train_sample_infos_label)
        train_data = tf.data.Dataset.from_tensor_slices((tf.constant(_train_sample_infos_path),
                                                         train_label_sparse))

        train_data = train_data.shuffle(buffer_size=10000).map(self._parse_fn).batch(BATCH_SIZE)

        # 加载验证数据
        val_label_sparse = self._label_to_sparse(_val_sample_infos_label)
        val_data = tf.data.Dataset.from_tensor_slices((tf.constant(_val_sample_infos_path),
                                                         val_label_sparse))

        val_data = val_data.map(self._parse_fn).batch(BATCH_SIZE)


        # 加载测试数据
        size = len(self._test_sample_infos[0]) - len(self._test_sample_infos[0])%BATCH_SIZE
        _test_sample_infos_path = self._test_sample_infos[0][:size]
        _test_sample_infos_label = self._test_sample_infos[1][:size]

        # 每一epoch测试集需要迭代的次数
        self.test_steps_per_epoch = len(_test_sample_infos_path) / BATCH_SIZE


        test_label_sparse = self._label_to_sparse(_test_sample_infos_label)
        test_data = tf.data.Dataset.from_tensor_slices((tf.constant(_test_sample_infos_path),
                                                         test_label_sparse))

        test_data = test_data.map(self._parse_fn).batch(BATCH_SIZE)

        return train_data, val_data, test_data

    def _label_to_sparse(self, label_index):
        label_index = np.array(label_index)
        indices = []
        value = []
        for i in range(label_index.shape[0]):
            for j in range(label_index.shape[1]):
                indices.append([i,j])
                value.append(label_index[i][j])
        return tf.SparseTensor(indices=indices, values=value, dense_shape=(label_index.shape[0], label_index.shape[1]))

    def _parse_fn(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = (tf.cast(img, tf.float32) / 127.5) - 1
        img = tf.image.resize(img, (IMAGE_HIGTH, IMAGE_WIDTH))
        return img, label

    def _is_source_data_complete(self):
        """
        Check if source data complete
        :return:
        """
        return \
            ops.exists(self._train_annotation_file_path)  \
            and ops.exists(self._test_annotation_file_path) and ops.exists(self._lexicon_file_path)

    def _init_dataset_sample_info(self):
        """
        organize dataset sample information, read all the lexicon information in lexicon list.
        Train, test, val sample information are lists like
        [(image_absolute_path_1, image_lexicon_index_1), (image_absolute_path_2, image_lexicon_index_2), ...]
        :return:
        """
        # # establish lexicon list
        # log.info('Start initialize lexicon information list...')
        # num_lines = sum(1 for _ in open(self._lexicon_file_path, 'r'))
        # with open(self._lexicon_file_path, 'r', encoding='utf-8') as file:
        #     for line in tqdm.tqdm(file, total=num_lines):
        #         self._lexicon_list.append(line.rstrip('\r').rstrip('\n'))


        ord_map_dict = CharDictBuilder.read_ord_map_dict(self._ord_map_dict_path)
        cache_dir = ops.join(self._dataset_dir, 'label', 'cache')
        if not ops.exists(cache_dir):
            os.mkdir(cache_dir)
        # establish train example info
        log.info('Start initialize train sample information list...')
        train_sample_infos_cache = ops.join(cache_dir, 'train_sample_infos_cache.pkl')
        if ops.exists(train_sample_infos_cache):
            with open(train_sample_infos_cache, 'rb') as fp:
                self._train_sample_infos = pickle.load(fp)
        else:
            num_lines = sum(1 for _ in open(self._train_annotation_file_path, 'r', encoding='utf-8', errors='ignore'))
            train_sample_infos_image_path = []
            train_sample_infos_label = []
            with open(self._train_annotation_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line in tqdm.tqdm(file, total=num_lines):

                    image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                    image_path = ops.join(self._dataset_dir, 'images', image_name)
                    label_index = [int(ord_map_dict[str(ord(label)) + '_ord']) for label in label_index]

                    if len(label_index)==10:
                        if not ops.exists(image_path):
                            raise ValueError('Example image {:s} not exist'.format(image_path))
                        train_sample_infos_image_path.append(image_path)
                        train_sample_infos_label.append(label_index)

                self._train_sample_infos += [train_sample_infos_image_path, train_sample_infos_label]
            with open(train_sample_infos_cache, 'wb') as fp:
                pickle.dump(self._train_sample_infos, fp)

        # establish test example info
        log.info('Start initialize testing sample information list...')
        test_sample_infos_cache = ops.join(cache_dir, 'test_sample_infos_cache.pkl')
        if ops.exists(test_sample_infos_cache):
            with open(test_sample_infos_cache, 'rb') as fp:
                self._test_sample_infos = pickle.load(fp)
        else:
            num_lines = sum(1 for _ in open(self._test_annotation_file_path, 'r', encoding='utf-8', errors='ignore'))
            test_sample_infos_image_path = []
            test_sample_infos_label = []
            with open(self._test_annotation_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                for line in tqdm.tqdm(file, total=num_lines):
                    image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                    image_path = ops.join(self._dataset_dir, 'images', image_name)
                    label_index = [int(ord_map_dict[str(ord(label)) + '_ord']) for label in label_index]

                    if len(label_index) == 10:
                        if not ops.exists(image_path):
                            raise ValueError('Example image {:s} not exist'.format(image_path))
                        test_sample_infos_image_path.append(image_path)
                        test_sample_infos_label.append(label_index)

                self._test_sample_infos += [test_sample_infos_image_path, test_sample_infos_label]
            with open(test_sample_infos_cache, 'wb') as fp:
                pickle.dump(self._test_sample_infos, fp)

    def _generate_char_dict(self):
        """
        generate the char dict and ord map dict json file according to the lexicon list.
        gather all the single characters used in lexicon list.
        :return:
        """
        char_lexicon_set = set()
        for lexcion in self._lexicon_list:
            for s in lexcion:
                char_lexicon_set.add(s)

        log.info('Char set length: {:d}'.format(len(char_lexicon_set)))

        char_lexicon_list = list(char_lexicon_set)
        char_dict_builder = CharDictBuilder()
        char_dict_builder.write_char_dict(char_lexicon_list, save_path=self._char_dict_path)
        char_dict_builder.map_ord_to_index(char_lexicon_list, save_path=self._ord_map_dict_path)

        log.info('Write char dict map complete')



class CharDictBuilder(object):
    """
        Build and read char dict
    """
    def __init__(self):
        pass

    @staticmethod
    def _read_chars(origin_char_list):
        """
        Read a list of chars or a file containing it.
        :param origin_char_list:
        :return:
        """
        if isinstance(origin_char_list, str):
            assert ops.exists(origin_char_list), \
                "Character list %s is not a file or could not be found" % origin_char_list
            with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
                chars = (l[0] for l in origin_f.readlines())
        elif isinstance(origin_char_list, Iterable):
            ok = all(map(lambda s: isinstance(s, str) and len(s) == 1, origin_char_list))
            assert ok, "Character list is not an Iterable of strings of length 1"
            chars = origin_char_list
        else:
            raise TypeError("Character list needs to be a file or a list of strings")
        return chars

    @staticmethod
    def _write_json(save_path, data):
        """

        :param save_path:
        :param data:
        :return:
        """
        if not save_path.endswith('.json'):
            raise ValueError('save path {:s} should be a json file'.format(save_path))
        os.makedirs(ops.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(data, json_f, sort_keys=True, indent=4)

    @staticmethod
    def write_char_dict(origin_char_list, save_path):
        """
        Writes the ordinal to char map used in int_to_char to decode predictions and labels.
        The file is read with CharDictBuilder.read_char_dict()
        :param origin_char_list: Either a path to file with character list, one a character per line, or a list or set
                                 of characters
        :param save_path: Destination file, full path.
        """
        char_dict = {str(ord(c)) + '_ord': c for c in CharDictBuilder._read_chars(origin_char_list)}
        CharDictBuilder._write_json(save_path, char_dict)

    @staticmethod
    def read_char_dict(dict_path):
        """

        :param dict_path:
        :return: a dict with ord(char) as key and char as value
        """
        with open(dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res

    @staticmethod
    def map_ord_to_index(origin_char_list, save_path):
        """
        Map ord of character in origin char list into index start from 0 in order to meet the output of the DNN
        :param origin_char_list:
        :param save_path:
        """
        ord_2_index_dict = {str(i) + '_index': str(ord(c)) for i, c in
                            enumerate(CharDictBuilder._read_chars(origin_char_list))}
        index_2_ord_dict = {str(ord(c)) + '_ord': str(i) for i, c in
                            enumerate(CharDictBuilder._read_chars(origin_char_list))}
        total_ord_map_index_dict = dict(ord_2_index_dict)
        total_ord_map_index_dict.update(index_2_ord_dict)
        CharDictBuilder._write_json(save_path, total_ord_map_index_dict)

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path):
        """

        :param ord_map_dict_path:
        :return:
        """
        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        return res

if __name__=='__main__':
    CS = ChineseString()
    a = CS.load_data()
    print(a)