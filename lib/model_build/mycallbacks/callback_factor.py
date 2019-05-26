import tensorflow as tf
import math
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras


class callback_factory():
    def __init__(self, config_list):
        self._config_list = config_list

    def load_callback(self):
        callbacks = [self._get_callback(callback_config) for callback_config in self._config_list]
        return callbacks

    def _get_callback(self, config):
        name = config['name']
        print('callback:使用{}'.format(name))
        if name == 'LearningRateScheduler':
            return self._LearningRateScheduler_init(config)
        if name == 'ModelCheckpoint':
            return self._ModelCheckpoint_init(config)
        if name == 'EarlyStopping':
            return self._EarlyStopping_init(config)
        if name == 'CSVLogger':
            return self._CSVLogger_init(config)
        if name == 'TensorBoard':
            return self._TensorBoard_init(config)
        else:
            raise KeyError('Unknown callback: {}'.format(name))

    def _LearningRateScheduler_init(self, config):
        self._lr = config['lr']
        self._epoch = config['epoch']
        schedule_name = config['schedule']
        print('学习率退火使用:{}'.format(schedule_name))
        if schedule_name == 'Cosine_Learning_Rate_Decay':
            schedule = self._Cosine_Learning_Rate_Decay
        return keras.callbacks.LearningRateScheduler(schedule=schedule)

    def _ModelCheckpoint_init(self, config):
        return keras.callbacks.ModelCheckpoint(config['filepath'],
                                           monitor=config['monitor'],
                                           verbose=config['verbose'],
                                           save_best_only=config['save_best_only'],
                                           save_weights_only=config['save_weights_only'],
                                           mode=config['mode'],
                                           period=config['period'])

    def _EarlyStopping_init(self, config):
        return keras.callbacks.EarlyStopping(monitor=config['monitor'],
                                               min_delta=config['min_delta'],
                                               patience=config['patience'],
                                               verbose=config['verbose'],
                                               mode=config['mode'],
                                               baseline=config['baseline'],
                                               restore_best_weights=config['restore_best_weights'])

    def _CSVLogger_init(self, config):
        return keras.callbacks.CSVLogger()

    def _TensorBoard_init(self, config):
        return keras.callbacks.TensorBoard(log_dir=config['log_dir'],
                                           histogram_freq=config['histogram_freq'],
                                           write_graph=config['write_graph'],
                                           write_images=config['write_images'],
                                           update_freq=config['update_freq'],
                                           profile_batch=config['profile_batch'])

    def _Cosine_Learning_Rate_Decay(self, epoch):
        new_lr = 0.5 * (1 + math.cos(epoch/self._epoch*math.pi))*self._lr
        print('epoch:{}, new_lr:{}'.format(epoch, new_lr))
        return new_lr