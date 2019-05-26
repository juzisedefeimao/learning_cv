import tensorflow as tf
from tensorflow.python import keras
import warnings
import numpy as np
import matplotlib.pyplot as plt
from cv_learning.lib.model_build.mylayers.mylayers import conv, depthwise_conv


layers = keras.layers

class darknet53(object):
    def __init__(self,include_top=True,
                 pooling='avg',
                 classes=1000,
                 **kwargs):
        self._include_top = include_top
        self._pooling = pooling
        self._classes = classes
        self._conv = conv
        self._conv_block = darknet_conv_block
        self._conv_block_big = darknet_conv_block_big

    def net(self, inputdata):
        """Instantiates the darknet53 architecture.

        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in your Keras config at `~/.keras/keras.json`.

        # Arguments
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization),
                  'imagenet' (pre-training on ImageNet),
                  or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
        # Determine proper input shape

        img_input = inputdata

        bn_axis = 3

        x = self._conv(32, (3,3),
                      stride=(1, 1),
                      padding='same',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      activition=layers.LeakyReLU(0.1),
                      name='conv1')(img_input)

        x = self._conv(64, (3, 3),
                      stride=(2, 2),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      activition=layers.LeakyReLU(0.1),
                      name='conv2')(x)


        x = self._conv_block_big(filters=[32, 64], kernel_size=(3, 3), strides=(1, 1), stage=1, conv_block_num=1)(x)

        x = self._conv(128, (3, 3),
                      stride=(2, 2),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      activition=layers.LeakyReLU(0.1),
                      name='conv3')(x)

        x = self._conv_block_big(filters=[64, 128], kernel_size=(3, 3), strides=(1, 1), stage=2, conv_block_num=2)(x)

        x = self._conv(256, (3, 3),
                      stride=(2, 2),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      activition=layers.LeakyReLU(0.1),
                      name='conv4')(x)

        x = self._conv_block_big(filters=[128, 256], kernel_size=(3, 3), strides=(1, 1), stage=3, conv_block_num=8)(x)

        x = self._conv(512, (3, 3),
                      stride=(2, 2),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      activition=layers.LeakyReLU(0.1),
                      name='conv5')(x)

        x = self._conv_block_big(filters=[256, 512], kernel_size=(3, 3), strides=(1, 1), stage=4, conv_block_num=8)(x)

        x = self._conv(1024, (3, 3),
                      stride=(2, 2),
                      padding='valid',
                      kernel_initializer='glorot_uniform',
                      is_bn=True,
                      bn_axis=bn_axis,
                      is_ZeroPadding=True,
                      activition=layers.LeakyReLU(0.1),
                      name='conv6')(x)

        x = self._conv_block_big(filters=[512, 1024], kernel_size=(3, 3), strides=(1, 1), stage=5, conv_block_num=4)(x)

        if self._include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(self._classes, activation='softmax', name='fc1000')(x)
        else:
            if self._pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif self._pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)
            elif self._pooling == None:
                x = x
            else:
                warnings.warn('The output shape of `darknet53(include_top=False)` '
                              'has been changed since Keras 2.2.0.')


        return x

class darknet53_depthwise(object):
    def __init__(self, include_top=True,
                 pooling='avg',
                 classes=1000,
                 **kwargs):
        self._include_top = include_top
        self._pooling = pooling
        self._classes = classes
        self._conv = conv
        self._depthwise_conv = depthwise_conv
        self._conv_block = darknet_conv_block
        self._conv_block_big = darknet_conv_block_big

    def net(self, inputdata):
        """Instantiates the darknet53 architecture.

        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in your Keras config at `~/.keras/keras.json`.

        # Arguments
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization),
                  'imagenet' (pre-training on ImageNet),
                  or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
        # Determine proper input shape

        img_input = inputdata

        bn_axis = 3

        x = self._conv(32, (3, 3),
                       stride=(1, 1),
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       activition=layers.LeakyReLU(0.1),
                       name='conv1')(img_input)

        x = self._depthwise_conv(64, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv2')(x)

        x = self._conv_block_big(filters=[32, 64], kernel_size=(3, 3), strides=(1, 1),
                                 stage=1, conv_block_num=1, depthwise=True)(x)

        x = self._depthwise_conv(128, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv3')(x)

        x = self._conv_block_big(filters=[64, 128], kernel_size=(3, 3), strides=(1, 1),
                                 stage=2, conv_block_num=2, depthwise=True)(x)

        x = self._depthwise_conv(256, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv4')(x)

        x = self._conv_block_big(filters=[128, 256], kernel_size=(3, 3), strides=(1, 1),
                                 stage=3, conv_block_num=8, depthwise=True)(x)

        x = self._depthwise_conv(512, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv5')(x)

        x = self._conv_block_big(filters=[256, 512], kernel_size=(3, 3), strides=(1, 1),
                                 stage=4, conv_block_num=8, depthwise=True)(x)

        x = self._depthwise_conv(1024, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv6')(x)

        x = self._conv_block_big(filters=[512, 1024], kernel_size=(3, 3), strides=(1, 1),
                                 stage=5, conv_block_num=4, depthwise=True)(x)

        if self._include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(self._classes, activation='softmax', name='fc1000')(x)
        else:
            if self._pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif self._pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)
            elif self._pooling == None:
                x = x
            else:
                warnings.warn('The output shape of `darknet53(include_top=False)` '
                              'has been changed since Keras 2.2.0.')

        return x

class darknet53_depthwisev2(object):
    def __init__(self, include_top=True,
                 pooling='avg',
                 classes=1000,
                 **kwargs):
        self._include_top = include_top
        self._pooling = pooling
        self._classes = classes
        self._conv = conv
        self._depthwise_conv = depthwise_conv
        self._conv_block = darknet_conv_block
        self._conv_block_big = darknet_conv_block_big

    def net(self, inputdata):

        img_input = inputdata

        bn_axis = 3

        x = self._conv(32, (3, 3),
                       stride=(1, 1),
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       activition=layers.LeakyReLU(0.1),
                       name='conv1')(img_input)

        x = self._conv(64, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv2')(x)
        x = self._conv(16, (1, 1),
                       stride=(1, 1),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv2_')(x)
        x = self._conv_block_big(filters=[64, 16], kernel_size=(3, 3), strides=(1, 1),
                                 stage=1, conv_block_num=1, depthwise=True)(x)

        x = self._conv(128, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv3')(x)
        x = self._conv(32, (1, 1),
                       stride=(1, 1),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv3_')(x)
        x = self._conv_block_big(filters=[128, 32], kernel_size=(3, 3), strides=(1, 1),
                                 stage=2, conv_block_num=2, depthwise=True)(x)

        x = self._conv(256, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv4')(x)
        x = self._conv(64, (1, 1),
                       stride=(1, 1),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv4_')(x)
        x = self._conv_block_big(filters=[256, 64], kernel_size=(3, 3), strides=(1, 1),
                                 stage=3, conv_block_num=8, depthwise=True)(x)

        x = self._conv(512, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv5')(x)
        x = self._conv(128, (1, 1),
                       stride=(1, 1),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv5_')(x)
        x = self._conv_block_big(filters=[512, 128], kernel_size=(3, 3), strides=(1, 1),
                                 stage=4, conv_block_num=8, depthwise=True)(x)

        x = self._conv(1024, (3, 3),
                       stride=(2, 2),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv6')(x)
        x = self._conv(256, (1, 1),
                       stride=(1, 1),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv6_')(x)
        x = self._conv_block_big(filters=[1024, 256], kernel_size=(3, 3), strides=(1, 1),
                                 stage=5, conv_block_num=4, depthwise=True)(x)
        x = self._conv(1024, (1, 1),
                       stride=(1, 1),
                       padding='valid',
                       kernel_initializer='glorot_uniform',
                       is_bn=True,
                       bn_axis=bn_axis,
                       is_ZeroPadding=True,
                       activition=layers.LeakyReLU(0.1),
                       name='conv7_')(x)

        if self._include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(self._classes, activation='softmax', name='fc1000')(x)
        else:
            if self._pooling == 'avg':
                x = layers.GlobalAveragePooling2D()(x)
            elif self._pooling == 'max':
                x = layers.GlobalMaxPooling2D()(x)
            elif self._pooling == None:
                x = x
            else:
                warnings.warn('The output shape of `darknet53(include_top=False)` '
                              'has been changed since Keras 2.2.0.')

        return x


class darknet_conv_block(layers.Layer):
    def __init__(self,filters,
                 kernel_size,
                 strides=(1, 1),
                 padding = 'same',
                 stage=None,
                 block=None,
                 **kwargs):
        super(darknet_conv_block, self).__init__(**kwargs)
        filters1, filters2 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        self.layer1 = conv(filters1, (1, 1),
                           stride=strides,
                           padding=padding,
                           kernel_initializer='glorot_uniform',
                           is_bn=True,
                           activition=layers.LeakyReLU(0.1),
                           name = conv_name_base + '2a')
        self.layer2 = conv(filters2, kernel_size,
                           stride=strides,
                           padding=padding,
                           kernel_initializer='glorot_uniform',
                           is_bn=True,
                           activition=layers.LeakyReLU(0.1),
                           name=conv_name_base + '2b')

    def call(self, inputs, **kwargs):
        """

                # Arguments
                    input_tensor: input tensor
                    kernel_size: default 3, the kernel size of
                        middle conv layer at main path
                    filters: list of integers, the filters of 3 conv layer at main path
                    stage: integer, current stage label, used for generating layer names
                    block: 'a','b'..., current block label, used for generating layer names
                    strides: Strides for the first conv layer in the block.

                # Returns
                    Output tensor for the block.

                Note that from stage 3,
                the first conv layer at main path is with strides=(2, 2)
                And the shortcut should have strides=(2, 2) as well
                """

        x = self.layer1(inputs)
        x = self.layer2(x)

        # 取和
        x = layers.add([x, inputs])
        x = layers.LeakyReLU(0.1)(x)
        return x

# 深度可分卷积的残差块
class darknet_conv_block_depthwise(layers.Layer):
    def __init__(self,filters,
                 kernel_size,
                 strides=(1, 1),
                 padding = 'same',
                 stage=None,
                 block=None,
                 **kwargs):
        super(darknet_conv_block_depthwise, self).__init__(**kwargs)
        filters1, filters2 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        self.layer1 = conv(filters1, (1, 1),
                           stride=strides,
                           padding=padding,
                           kernel_initializer='glorot_uniform',
                           is_bn=True,
                           activition=layers.LeakyReLU(0.1),
                           name = conv_name_base + '2a')
        self.layer2 = depthwise_conv(filters2, kernel_size,
                           stride=strides,
                           padding=padding,
                           kernel_initializer='glorot_uniform',
                           is_bn=True,
                           activition=layers.LeakyReLU(0.1),
                           name=conv_name_base + '2b')

    def call(self, inputs, **kwargs):
        """

                # Arguments
                    input_tensor: input tensor
                    kernel_size: default 3, the kernel size of
                        middle conv layer at main path
                    filters: list of integers, the filters of 3 conv layer at main path
                    stage: integer, current stage label, used for generating layer names
                    block: 'a','b'..., current block label, used for generating layer names
                    strides: Strides for the first conv layer in the block.

                # Returns
                    Output tensor for the block.

                Note that from stage 3,
                the first conv layer at main path is with strides=(2, 2)
                And the shortcut should have strides=(2, 2) as well
                """

        x = self.layer1(inputs)
        x = self.layer2(x)

        # 取和
        x = layers.add([x, inputs])
        x = layers.LeakyReLU(0.1)(x)
        return x

# 多个残差块组成的大块，可以选者是否深度可分
class darknet_conv_block_big(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1),
                 padding = 'same', stage=None, conv_block_num=None, depthwise=False, **kwargs):
        super(darknet_conv_block_big, self).__init__(name='conv_block_big' + str(stage))
        # if depthwise:
        #     self._conv_block = darknet_conv_block_depthwise
        # else:
        #     self._conv_block = darknet_conv_block
        self.conv_block_init(filters, kernel_size, strides = strides, padding=padding,
                             stage=stage, conv_block_num=conv_block_num, depthwise=depthwise)

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(len(self.layer)):
            x = self.layer[i](x)
        return x

    def conv_block_init(self, filters, kernel_size, strides=(1, 1),
                 padding = 'same', stage=None, conv_block_num=None, depthwise=False):
        self.layer = []
        if depthwise:
            conv_block = darknet_conv_block_depthwise
        else:
            conv_block = darknet_conv_block
        for i in range(conv_block_num):
            self.layer.append(conv_block(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                         stage=stage, block=chr(ord('a') + i)))



