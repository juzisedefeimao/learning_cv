# from tensorflow.python import keras
import tensorflow.keras as keras


class optimizers_factory():
    def __init__(self, optimizer_config):
        self._config = optimizer_config

    def load_optimizer(self):
        my_optimizer = self._get_optimizer(self._config)
        return my_optimizer

    def _get_optimizer(self, config):
        name = config['name']
        print('优化器:使用{}'.format(name))
        if name=='SGD':
            return self._SGD_init(config)
        if name=='Adam':
            return self._Adam_init(config)
        else:
            raise KeyError('没有所选择的优化器{}'.format(name))

    def _SGD_init(self, config):
        lr = config['lr']
        momentum = config['momentum']
        return keras.optimizers.SGD(lr = lr, momentum=momentum)

    def _Adam_init(self, config):
        lr = config['lr']
        return keras.optimizers.Adam(lr=lr)