import tensorflow as tf
from tensorflow.python import keras
import warnings
layers = keras.layers

# 权重初始化方:'he_normal'\'glorot_uniform'
class conv(layers.Layer):
    def __init__(self,filter, kernel,
                 stride=(1,1),
                 padding='same',
                 kernel_initializer=None,
                 is_bn=True,
                 bn_axis=3,
                 is_ZeroPadding=False,
                 ZeroPadding=((1, 0), (1, 0)),
                 activition=None,
                 name=None,
                 **kwargs):
        super(conv, self).__init__(name=name)
        self._is_bn = is_bn
        self._padding = padding
        self._is_ZeroPadding = is_ZeroPadding
        self._ZeroPadding = ZeroPadding
        self._activition = activition

        self._conv_layer = layers.Conv2D(filter, kernel,
                          strides=stride,
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          name='conv_' + name)
        if self._is_bn:
            self._bn_layer = layers.BatchNormalization(axis=bn_axis, name='bn_' + name)

    def call(self, inputs, **kwargs):
        x = inputs
        if self._is_ZeroPadding and self._padding=='valid':
            x = layers.ZeroPadding2D(padding=self._ZeroPadding)(x)
        x = self._conv_layer(x)
        if self._is_bn:
            x = self._bn_layer(x)
        if self._activition is not None:
            if isinstance(self._activition, str):
                x = layers.Activation(self._activition)(x)
            else:
                x = self._activition(x)
        else:
            x = layers.Activation('relu')(x)
        return x

# 深度可分离卷积
class depthwise_conv(layers.Layer):
    def __init__(self,filter, kernel,
                 stride=(1,1),
                 padding='same',
                 kernel_initializer=None,
                 is_bn=True,
                 bn_axis=3,
                 is_ZeroPadding=False,
                 ZeroPadding=((1, 0), (1, 0)),
                 activition=None,
                 name=None,
                 **kwargs):
        super(depthwise_conv, self).__init__(name=name)
        self._is_bn = is_bn
        self._padding = padding
        self._is_ZeroPadding = is_ZeroPadding
        self._ZeroPadding = ZeroPadding
        self._activition = activition

        self._depthwise_layer = layers.DepthwiseConv2D(kernel,
                                                       strides=stride,
                                                       padding=padding,
                                                       depth_multiplier=1,
                                                       use_bias=False,
                                                       depthwise_initializer='glorot_uniform',
                                                       bias_initializer='zeros',
                                                       name='depthwise_conv_' + name)
        self._conv_layer = layers.Conv2D(filter, kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='valid',
                          kernel_initializer=kernel_initializer,
                          name='conv_' + name)

        if self._is_bn:
            self._bn_layer1 = layers.BatchNormalization(axis=bn_axis, name='depthwise_bn_' + name)
            self._bn_layer2 = layers.BatchNormalization(axis=bn_axis, name='bn_' + name)

    def call(self, inputs, **kwargs):
        x = inputs
        if self._is_ZeroPadding and self._padding=='valid':
            x = layers.ZeroPadding2D(padding=self._ZeroPadding)(x)

        x = self._depthwise_layer(x)

        if self._is_bn:
            x = self._bn_layer1(x)
        if self._activition is not None:
            if isinstance(self._activition, str):
                x = layers.Activation(self._activition)(x)
            else:
                x = self._activition(x)
        else:
            x = layers.Activation('relu')(x)

        x = self._conv_layer(x)

        if self._is_bn:
            x = self._bn_layer2(x)
        if self._activition is not None:
            if isinstance(self._activition, str):
                x = layers.Activation(self._activition)(x)
            else:
                x = self._activition(x)
        else:
            x = layers.Activation('relu')(x)
        return x

# 可变形卷积
class deform_conv(layers.Layer):
    def __init__(self,filter, kernel,
                 stride=(1,1),
                 padding='same',
                 kernel_initializer=None,
                 is_bn=True,
                 bn_axis=3,
                 is_ZeroPadding=False,
                 ZeroPadding=((1, 0), (1, 0)),
                 activition=None,
                 name=None,
                 **kwargs):
        super(deform_conv, self).__init__(name=name)
        self.is_bn = is_bn
        self.padding = padding
        self.is_ZeroPadding = is_ZeroPadding
        self.ZeroPadding = ZeroPadding
        self.activition = activition

        self.conv_layer = layers.Conv2D(filter, kernel,
                          strides=stride,
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          name='conv_' + name)
        if self.is_bn:
            self.bn_layer = layers.BatchNormalization(axis=bn_axis, name='bn_' + name)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.is_ZeroPadding and self.padding=='valid':
            x = layers.ZeroPadding2D(padding=self.ZeroPadding)(x)
        x = self.conv_layer(x)
        if self.is_bn:
            x = self.bn_layer(x)
        if self.activition is not None:
            if isinstance(self.activition, str):
                x = layers.Activation(self.activition)(x)
            else:
                x = self.activition(x)
        else:
            x = layers.Activation('relu')(x)
        return x

# 空洞卷积
class dilated_conv(layers.Layer):
    def __init__(self,filter, kernel,
                 stride=(1,1),
                 padding='same',
                 kernel_initializer=None,
                 is_bn=True,
                 bn_axis=3,
                 is_ZeroPadding=False,
                 ZeroPadding=((1, 0), (1, 0)),
                 activition=None,
                 name=None,
                 **kwargs):
        super(dilated_conv, self).__init__(name=name)
        self.is_bn = is_bn
        self.padding = padding
        self.is_ZeroPadding = is_ZeroPadding
        self.ZeroPadding = ZeroPadding
        self.activition = activition

        self.conv_layer = layers.Conv2D(filter, kernel,
                          strides=stride,
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          name='conv_' + name)
        if self.is_bn:
            self.bn_layer = layers.BatchNormalization(axis=bn_axis, name='bn_' + name)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.is_ZeroPadding and self.padding=='valid':
            x = layers.ZeroPadding2D(padding=self.ZeroPadding)(x)
        x = self.conv_layer(x)
        if self.is_bn:
            x = self.bn_layer(x)
        if self.activition is not None:
            if isinstance(self.activition, str):
                x = layers.Activation(self.activition)(x)
            else:
                x = self.activition(x)
        else:
            x = layers.Activation('relu')(x)
        return x