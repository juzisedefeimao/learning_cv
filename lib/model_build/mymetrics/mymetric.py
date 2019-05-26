from cv_learning.lib.model_build.myloss.myloss import CTC_Loss
import tensorflow as tf
from tensorflow.python import keras

class crnn_metric(keras.metrics.Metric):
    def __init__(self, name='crnn_metric', **kwargs):
        super(crnn_metric, self).__init__(name=name, **kwargs)
        self.true_postives = self.add_weight(name='tp', initializer='zeros')
        self._num = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = CTC_Loss()(y_true, y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            loss = tf.multiply(sample_weight, loss)
        self._num = self._num + 1
        return self.true_postives.assign_add(loss)

    def result(self):
        res = self.true_postives / self._num
        res = tf.identity(res)
        return res

    def reset_states(self):
        self._num = 0
        self.true_postives.assign(0.)