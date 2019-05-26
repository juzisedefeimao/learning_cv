from cv_learning.lib.datasets.ChineseString import ChineseString
from cv_learning.lib.datasets.ILSVRC2017 import ILSVRC2017
from cv_learning.lib.datasets.tensorflow_datasets import cifar10, cifar100

import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

class datasets_factory():
    def __init__(self, datasets_config):
        self._config = datasets_config

    def load_datasets(self):
        my_datasets = self._get_dataset(self._config)
        return my_datasets

    def _get_dataset(self, config):
        name = config['name']
        print('数据集:使用{}'.format(name))
        if name == 'ChineseString':
            return self._ChineseString_init(config)
        if name == 'ILSVRC2017':
            return self._ILSVRC2017_init(config)
        if name == 'cifar10':
            return self._cifar10_init(config)
        if name == 'cifar100':
            return self._cifar100_init(config)
        else:
            raise KeyError('Unknown dataset: {}'.format(name))

    def _ChineseString_init(self, config):
        return ChineseString()

    def _ILSVRC2017_init(self, config):
        return ILSVRC2017()

    def _cifar10_init(self, config):
        return cifar10(validation_split=config['validation_split'],
                       batch_size=config['batch_size'],
                       shuffle=config['shuffle'])

    def _cifar100_init(self, config):
        return cifar100(validation_split=config['validation_split'],
                       batch_size=config['batch_size'],
                       shuffle=config['shuffle'])

    def _data_augmentation(self, aug_config):
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=0,
            width_shift_range=0.,
            height_shift_range=0.,
            brightness_range=None,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format='channels_last',
            validation_split=0.0,
            dtype='float32')
        return datagen