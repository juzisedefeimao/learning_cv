import tensorflow as tf
import random
import math
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras


class cifar10():
    def __init__(self, validation_split=0.1, batch_size=32, shuffle=128):
        self._split = validation_split
        self._batch_size = batch_size
        self._shuffle = shuffle

    def load_data_(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        # 图片、标签预处理
        x_train, y_train = data_tool.data_preprossing((x_train, y_train))
        x_test, y_test = data_tool.data_preprossing((x_test, y_test))

        # 从训练集里切分验证集
        (x_train, y_train),(x_val, y_val) = data_tool.validation_split((x_train, y_train), split=self._split)


        # 准备数据
        train_data = data_tool.df_to_dataset((x_train,y_train),
                                             batch_size=self._batch_size, buffer_size=self._shuffle)

        val_data = data_tool.df_to_dataset((x_val, y_val), shuffle=False,
                                           batch_size=self._batch_size, buffer_size=self._shuffle)

        test_data = data_tool.df_to_dataset((x_test, y_test), shuffle=False,
                                            batch_size=self._batch_size, buffer_size=self._shuffle)

        return train_data, val_data, test_data

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        datagen_train = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            channel_shift_range=20,
            horizontal_flip=True,
            vertical_flip=True)
        datagen_val = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)

        datagen_test = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)


        # 从训练集里切分验证集
        (x_train, y_train),(x_val, y_val) = data_tool.validation_split((x_train, y_train), split=self._split)

        datagen_train.fit(x_train)
        train_data = datagen_train.flow(x_train, y_train, batch_size=self._batch_size)

        datagen_val.fit(x_val)
        val_data = datagen_val.flow(x_val, y_val, batch_size=self._batch_size, shuffle=False)

        datagen_test.fit(x_test)
        test_data = datagen_test.flow(x_test, y_test, batch_size=self._batch_size, shuffle=False)


        return train_data, val_data, test_data


class cifar100():
    def __init__(self, validation_split=0.1, batch_size=32, shuffle=128):
        self._split = validation_split
        self._batch_size = batch_size
        self._shuffle = shuffle

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

        # 图片、标签预处理
        x_train, y_train = data_tool.data_preprossing((x_train, y_train))
        x_test, y_test = data_tool.data_preprossing((x_test, y_test))

        # 从训练集里切分验证集
        (x_train, y_train), (x_val, y_val) = data_tool.validation_split((x_train, y_train), split=self._split)

        # 准备数据
        train_data = data_tool.df_to_dataset((x_train, y_train),
                                             batch_size=self._batch_size, buffer_size=self._shuffle)

        val_data = data_tool.df_to_dataset((x_val, y_val), shuffle=False,
                                           batch_size=self._batch_size, buffer_size=self._shuffle)

        test_data = data_tool.df_to_dataset((x_test, y_test), shuffle=False,
                                            batch_size=self._batch_size, buffer_size=self._shuffle)

        return train_data, val_data, test_data

class data_tool():

    @staticmethod
    def df_to_dataset(dataframe, shuffle=True, batch_size=32, buffer_size=128):
        ds = tf.data.Dataset.from_tensor_slices((dataframe[0], dataframe[1]))
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.batch(batch_size)
        return ds

    @staticmethod
    def validation_split(train_data, split=0.1):
        x_train_ = train_data[0]
        y_train_ = train_data[1]
        data_num = len(x_train_)
        x_train = x_train_[0:int(data_num * (1-split))]
        x_val = x_train_[int(data_num * (1-split)):]
        y_train = y_train_[0:int(data_num * (1-split))]
        y_val = y_train_[int(data_num * (1-split)):]
        return (x_train, y_train),(x_val, y_val)

    @staticmethod
    def data_preprossing(data):
        x_data = data[0].astype('float32') / 255
        y_data = data[1].astype('int32')
        return x_data, y_data

    @staticmethod
    def preprossing(data):
        x_data = data.astype('float32') / 255
        return x_data
