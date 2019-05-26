from cv_learning.lib.model_build.myloss.myloss import *
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

class loss_factory():
    def __init__(self, loss_config):
        self._config = loss_config

    def load_loss(self):
        my_loss = self._get_loss(self._config)
        return my_loss

    def _get_loss(self, config):
        name = config['name']
        print('损失函数:使用{}'.format(name))

        if name=='ctc_loss':
            return CTC_Loss()
        if name == 'sparse_categorical_crossentropy':
            return SparseCategoricalCrossentropy(label_smooth_c=config['label_smoothing'])
        if name=='yolo_loss':
            return YOLO_Loss()
        else:
            raise KeyError('没有所选择的损失函数{}'.format(name))