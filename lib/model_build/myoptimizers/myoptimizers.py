import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
K = keras.backend

class SGD_noise(keras.optimizers.Optimizer):
    """
    每次更新梯度都会加入一个随机高斯噪声，以增加模型的鲁棒性
    """

class myoptimizers(keras.optimizers.Optimizer):
    """Keras中简单自定义SGD优化器
        每隔一定的batch才更新一次参数
        """
    def __init__(self,lr = 0.01, steps_per_update = 1,** kwargs):
        super(myoptimizers, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype = 'int64', name = 'iterations')
            self.lr = K.variable(lr,name = 'lr')
            self.steps_per_update = steps_per_update

    def get_updates(self,loss, params):
        """主要的参数更新算法
        """
        shapes = [K.int_shape(p) for p in params]
        sum_grads = [K.zeros(shape) for shape in shapes]
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations] + sum_grads
        for p, g, sg in zip(params, grads, sum_grads):
            new_p = p - self.lr * sg / float(self.steps_per_update)
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
                cond = K.equal(self.iterations % self.steps_per_update, 0)
                #  满足条件才更新参数
                self.updates.append(K.switch(cond, K.update(p, new_p), p))
                #  满足条件就要重新累积，不满足条件直接累积
                self.updates.append(K.switch(cond, K.update(sg, g), K.update(sg, sg + g)))
        return self.updates

    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),
                  'steps_per_update': self.steps_per_update}
        base_config = super(myoptimizers, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))