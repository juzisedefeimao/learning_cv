import tensorflow as tf
from tensorflow.python import keras
import warnings
import numpy as np
import matplotlib.pyplot as plt
from cv_learning.lib.model_build.mymodels.darknet import *
from cv_learning.lib.model_build.mylayers.mylayers import *


layers = keras.layers

class crnn_net():
    def __init__(self):
        self.conv = conv
        self.conv_block = darknet_conv_block
        self.conv_block_big = darknet_conv_block_big

    def net(self, inputdata):
        x = self._feature_sequence_extraction(inputdata)
        x = self._map_to_sequence(x)
        x = self._sequence_label(x)
        return x

    # 原文里的卷积结构
    def _feature_sequence_extraction_(self, inputdata):
        bn_axis = 3
        x = self.conv(64, (3, 3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition='relu',
                      name='conv1')(inputdata)
        x = layers.MaxPool2D((2, 2), strides=(2,2), name='maxpool1')(x)
        x = self.conv(128, (3, 3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition='relu',
                      name='conv2')(x)
        x = layers.MaxPool2D((2, 2), strides=(2, 2), name='maxpool2')(x)
        x = self.conv(256, (3, 3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition='relu',
                      name='conv3')(x)
        x = self.conv(256, (3, 3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition='relu',
                      name='conv4')(x)
        x = layers.MaxPool2D((2, 1), strides=(2, 1), name='maxpool3')(x)
        x = self.conv(512, (3, 3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition='relu',
                      name='conv5')(x)
        x = self.conv(512, (3, 3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition='relu',
                      name='conv6')(x)
        x = layers.MaxPool2D((2, 1), strides=(2, 1), name='maxpool4')(x)
        x = self.conv(512, (2, 2),
                      stride=(1, 1),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      ZeroPadding=((0, 0), (1, 0)),
                      activition='relu',
                      name='conv7')(x)
        return x

    # 基于darknet53
    def _feature_sequence_extraction(self, inputdata):
        bn_axis = 3
        x = self.conv(32, (3, 3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition=layers.advanced_activations.LeakyReLU(0.1),
                      name='conv1')(inputdata)

        x = self.conv(64, (3, 3),
                      stride=(2, 2),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      activition=layers.advanced_activations.LeakyReLU(0.1),
                      name='conv2')(x)

        x = self.conv_block_big(filters=[32, 64], kernel_size=(3, 3), strides=(1, 1), stage=1, conv_block_num=1)(x)

        x = self.conv(128, (3, 3),
                      stride=(2, 2),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      activition=layers.advanced_activations.LeakyReLU(0.1),
                      name='conv3')(x)

        x = self.conv_block_big(filters=[64, 128], kernel_size=(3, 3), strides=(1, 1), stage=2, conv_block_num=2)(x)

        x = self.conv(256, (3, 3),
                      stride=(2, 1),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      ZeroPadding=((1,0),(1,1)),
                      activition=layers.advanced_activations.LeakyReLU(0.1),
                      name='conv4')(x)

        x = self.conv_block_big(filters=[128, 256], kernel_size=(3, 3), strides=(1, 1), stage=3, conv_block_num=8)(x)

        x = self.conv(512, (3, 3),
                      stride=(2, 1),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      ZeroPadding=((1, 0), (1, 1)),
                      activition=layers.advanced_activations.LeakyReLU(0.1),
                      name='conv5')(x)

        x = self.conv_block_big(filters=[256, 512], kernel_size=(2, 2), strides=(1, 1), stage=4, conv_block_num=8)(x)
        x = self.conv(512, (2, 2),
                      stride=(1, 1),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      ZeroPadding=((0,0),(1,0)),
                      activition=layers.advanced_activations.LeakyReLU(0.1),
                      name='conv6')(x)

        return x

    def _map_to_sequence(self, inputdata):
        x = tf.squeeze(inputdata, axis=1)
        return x

    def _sequence_label(self, inputdata):
        x = layers.Bidirectional(layers.CuDNNLSTM(256, return_sequences=True), merge_mode='concat')(inputdata)
        x = layers.Bidirectional(layers.CuDNNLSTM(256, return_sequences=True), merge_mode='concat')(x)
        # x = layers.Dropout(0.5)(x)
        x = layers.Dense(5825, activation='softmax', use_bias=False)(x)


        return x