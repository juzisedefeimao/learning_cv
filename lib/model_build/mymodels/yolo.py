from cv_learning.lib.model_build.mylayers.mylayers import *
from cv_learning.lib.model_build.mymodels.darknet import darknet_conv_block_big

import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

layers = keras.layers

class yolo():
    def __init__(self, num_anchors=9, classes=20):
        self._conv = conv
        self._conv_block_big = darknet_conv_block_big
        self._darknet_conv_set = darknet_conv_set
        self._darknet_predict = darknet_predict
        self._darknet_upsampling = darknet_upsampling
        self._num_anchors = num_anchors
        self._classes = classes

    def net(self, inputdata):
        img_input = inputdata

        bn_axis = 3

        x = self._conv(32, (3, 3),
                       stride=(1, 1),
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       activition=keras.layers.LeakyReLU(0.1),
                       name='conv1')(img_input)

        x = self._conv(64, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=keras.layers.LeakyReLU(0.1),
                       name='conv2')(x)

        x = self._conv_block_big(filters=[32, 64], kernel_size=(3, 3), strides=(1, 1), stage=1, conv_block_num=1)(x)

        x = self._conv(128, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=keras.layers.LeakyReLU(0.1),
                       name='conv3')(x)

        x = self._conv_block_big(filters=[64, 128], kernel_size=(3, 3), strides=(1, 1), stage=2, conv_block_num=2)(x)

        x = self._conv(256, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=keras.layers.LeakyReLU(0.1),
                       name='conv4')(x)

        p1 = self._conv_block_big(filters=[128, 256], kernel_size=(3, 3), strides=(1, 1), stage=3, conv_block_num=8)(x)

        x = self._conv(512, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=keras.layers.LeakyReLU(0.1),
                       name='conv5')(p1)

        p2 = self._conv_block_big(filters=[256, 512], kernel_size=(3, 3), strides=(1, 1), stage=4, conv_block_num=8)(x)

        x = self._conv(1024, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=keras.layers.LeakyReLU(0.1),
                       name='conv6')(p2)

        x = self._conv_block_big(filters=[512, 1024], kernel_size=(3, 3), strides=(1, 1), stage=5, conv_block_num=4)(x)

        x = self._darknet_conv_set(512, name='conv_set1')(x)
        y1 = self._darknet_predict(1024, (self._num_anchors//3)*(self._classes + 5), name='predict1')(x)

        x = self._darknet_upsampling(256)(x)
        x = keras.layers.Concatenate()([x, p2])
        x = self._darknet_conv_set(256, name='conv_set2')(x)
        y2 = self._darknet_predict(512, (self._num_anchors//3)*(self._classes + 5), name='predict2')(x)

        x = self._darknet_upsampling(128)(x)
        x = keras.layers.Concatenate()([x, p1])
        x = self._darknet_conv_set(128, name='conv_set3')(x)
        y3 = self._darknet_predict(256, (self._num_anchors//3)*(self._classes + 5), name='predict3')(x)

        return [y1, y2, y3]

class darknet_conv_set(layers.Layer):
    def __init__(self, filter,
                 padding='same',
                 **kwargs):
        super(darknet_conv_set, self).__init__(**kwargs)
        self.conv_init(filter=filter, padding=padding)

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(len(self._layer)):
            x = self._layer[i](x)
        return x

    def conv_init(self, filter,padding = 'same'):
        self._layer = []
        for i in range(5):
            if i%2==0:
                filter_ = filter
                kernel = (1, 1)
            else:
                filter_ = filter * 2
                kernel = (3, 3)
            self._layer.append(conv(filter_, kernel,
                           stride=(1, 1),
                           padding=padding,
                           kernel_initializer='glorot_uniform',
                           is_bn=True,
                           activition=keras.layers.LeakyReLU(0.1),
                           name='darknet_conv_set_'+str(i))
                              )

class darknet_predict(layers.Layer):
    def __init__(self, filters, out_dim, padding = 'same', **kwargs):
        super(darknet_predict, self).__init__(**kwargs)
        self._layer1 = conv(filters, (3, 3),
                             stride=(1, 1),
                             padding=padding,
                             kernel_initializer='glorot_uniform',
                             is_bn=True,
                             activition=keras.layers.LeakyReLU(0.1),
                             name='darknet_predict_1')
        self._layer2 = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1))

    def call(self, inputs, **kwargs):
        x = self._layer1(inputs)
        x = self._layer2(x)
        return x

class darknet_upsampling(layers.Layer):
    def __init__(self, filters, padding = 'same', **kwargs):
        super(darknet_upsampling, self).__init__(**kwargs)
        self._layer1 = conv(filters, (1, 1),
                             stride=(1, 1),
                             padding=padding,
                             kernel_initializer='glorot_uniform',
                             is_bn=True,
                             activition=keras.layers.LeakyReLU(0.1),
                             name='darknet_upsampling_conv1')
        self._layer2 = keras.layers.UpSampling2D()

    def call(self, inputs, **kwargs):
        x = self._layer1(inputs)
        x = self._layer2(x)
        return x