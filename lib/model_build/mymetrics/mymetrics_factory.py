from cv_learning.lib.model_build.mymetrics.mymetric import crnn_metric
from tensorflow.python import keras

class metrics_factory():
    def __init__(self, metric_config):
        self._config = metric_config

    def load_metric(self):
        my_metric = self._get_metric(self._config)
        return my_metric

    def _get_metric(self, config):
        name = config['name']
        print('评估方法:使用{}'.format(name))
        if name=='crnn_metric':
            return crnn_metric()
        if name == 'SparseCategoricalAccuracy':
            return keras.metrics.SparseCategoricalAccuracy()
        else:
            raise KeyError('没有所选择的评估函数{}'.format(name))