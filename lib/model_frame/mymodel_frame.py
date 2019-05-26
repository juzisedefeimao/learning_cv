from cv_learning.lib.datasets.mydatasets_factory import datasets_factory
from cv_learning.lib.model_build.mymodels.mymodels_factory import models_factory
from cv_learning.lib.model_build.myoptimizers.myoptimizers_factory import optimizers_factory
from cv_learning.lib.model_build.mymetrics.mymetrics_factory import metrics_factory
from cv_learning.lib.model_build.mycallbacks.callback_factor import callback_factory
from cv_learning.lib.model_build.myloss.myloss_factory import loss_factory
import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
import os
import yaml




class my_frame():
    def __init__(self, yaml_dir = 'D:\\jjj\\GitHub\\cv_learning\\lib\\config\\config.yaml'):

        self._config = self._read_config(yaml_dir)

        net_name = self._config.get('use_net')
        self._model_name = net_name

        self._save_model_dir = os.path.join(self._config['save_model_dir'], self._model_name)
        if not os.path.exists(self._save_model_dir):
            os.makedirs(self._save_model_dir)
        print('保存模型路径', self._save_model_dir)

        self._save_log_dir = os.path.join(self._config['save_log_dir'], self._model_name)
        if not os.path.exists(self._save_log_dir):
            os.makedirs(self._save_log_dir)
        print('保存log路径', self._save_log_dir)

        # ============================加载net类====================================
        # net的工厂类，
        net_wrapper = models_factory(self._get_net_config())
        # 从net工厂类内加载所需要的net，得到所需net的类
        self._my_net, self._input_shape, self._label_shape = net_wrapper.load_net()

        # ============================加载数据集类====================================
        # 数据集的工厂类
        datasets_wrapper = datasets_factory(self._get_datasets_config())
        # 从数据集的工厂类中加载所需要的数据集类
        self._my_datasets = datasets_wrapper.load_datasets()

        # ==========================加载优化器类========================================
        # 优化器的工厂类
        optimizer_wrapper = optimizers_factory(self._get_optimizer_config())
        # 从优化器的工厂类中加载所需要的优化器
        self._my_optimizer = optimizer_wrapper.load_optimizer()

        # ===============================加载评估方法类================================
        # 评估方法的工厂类
        metric_wrapper = metrics_factory(self._get_metric_config())
        # 从评估方法的工厂类中加载说需要的评估方法
        self._my_metric = metric_wrapper.load_metric()

        # ===============================加载callback方法类================================
        # callback方法的工厂类
        callback_wrapper = callback_factory(self._get_callback_config())
        # 从评估方法的工厂类中加载说需要的评估方法
        self._my_callback = callback_wrapper.load_callback()

        # =================================加载损失函数====================================
        # 损失函数的工厂类
        loss_wrapper = loss_factory(self._get_loss_config())
        # 从损失函数的工厂类中加载所需要的损失函数
        self._my_loss = loss_wrapper.load_loss()




    # ================================得到各个组件的配置=============================================
    # 加载所有的配置文件
    def _read_config(self, yaml_dir):
        with open(yaml_dir, 'rb') as fp:
            config = fp.read()

        config = yaml.load(config)
        return config

    # 读取所用net的配置
    def _get_net_config(self):
        net_name = self._config.get('use_net')
        net_config = self._config.get('net')
        return net_config[net_name]

    # 读取所用数据集的配置
    def _get_datasets_config(self):
        datasets_name = self._config.get('use_datasets')
        datasets_config = self._config.get('datasets')
        return datasets_config[datasets_name]

    # 读取所用损失函数的配置
    def _get_loss_config(self):
        loss_name = self._config.get('use_loss')
        loss_config = self._config.get('loss')
        return loss_config[loss_name]

    # 读取所用的优化器的配置
    def _get_optimizer_config(self):
        optimizer_name = self._config.get('use_optimizer')
        optimizer_config = self._config.get('optimizer')
        self._lr = optimizer_config[optimizer_name]['lr']
        return optimizer_config[optimizer_name]

    # 读取所用评估的配置
    def _get_metric_config(self):
        metric_name = self._config.get('use_metric')
        metric_config = self._config.get('metric')
        return metric_config[metric_name]

    # 读取所用回调的配置
    def _get_callback_config(self):
        callback_name_list = self._config.get('use_callback')
        callback_config = self._config.get('callback')
        callback_config_list = []
        for name in callback_name_list:
            if name=='ModelCheckpoint':
                filename = self._model_name + '-weight_{epoch}'
                save_root = os.path.join(self._save_model_dir, filename)
                MC_config = callback_config[name]
                MC_config['filepath'] = save_root
                callback_config_list.append(MC_config)
            elif name=='TensorBoard':
                filename = self._model_name + '-log'
                save_root = os.path.join(self._save_log_dir, filename)
                train_log_root = os.path.join(save_root, 'train')
                if not os.path.exists(train_log_root):
                    os.makedirs(train_log_root)
                TB_config = callback_config[name]
                TB_config['log_dir'] = save_root
                callback_config_list.append(TB_config)
            elif name == 'LearningRateScheduler':
                LRS_config = callback_config[name]
                LRS_config['lr'] = self._lr
                LRS_config['epoch'] = self._config['epoch']
                callback_config_list.append(LRS_config)
            else:
                callback_config_list.append(callback_config[name])
        # print('kkk', callback_config_list)
        return callback_config_list

    # ===========================================建造、训练、评估、预测==================================
    # 建造模型
    def build_model(self):
        inputdata = keras.layers.Input(self._input_shape)
        x = self._my_net.net(inputdata)
        model = keras.models.Model(inputs=inputdata, outputs=x)
        return model

    def train(self):
        label = keras.Input((*self._label_shape,))
        train_data, val_data, test_data = self.select_datasets()
        model = self.build_model()
        optimizer = self.select_optimier()
        loss = self.select_loss()
        metric = self.select_metric()
        callbacks = self.select_callback()
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[metric],
                      target_tensors=label
                      )

        # 加载模型参数
        init_epoch = self._load_model(model)
        try:
            model.fit_generator(train_data,
                                  validation_data=val_data,
                                  epochs=self._config['epoch'],
                                  initial_epoch=init_epoch,
                                  callbacks=callbacks)
        except:
            model.fit(train_data,
                        validation_data=val_data,
                        epochs=self._config['epoch'],
                        initial_epoch=init_epoch,
                        callbacks=callbacks
                        )

        model.evaluate(test_data)

    # 利用训练集训练模型
    def train_(self):
        train_data, val_data, test_data = self.select_datasets()
        model = self.build_model()
        optimizer = self.select_optimier()
        loss = self.select_loss()
        # train_metric, val_metric = self.select_metric()

        epoch = self._load_model(model)

        for epoch in range(60)[epoch:]:
            print('Start of epoch %d' % (epoch,))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_data):
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train)
                    loss_value = loss(y_batch_train, logits)
                    grads = tape.gradient(loss_value, model.trainable_variables)
                    print('步数:step', loss_value)
                    # print('label', y_batch_train)
                    # print('logist', logits)
                    # 限制梯度更新不要过大
                    clip_grad = False
                    if clip_grad:
                        grads, norm = tf.clip_by_global_norm(grads, 10)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Update training metric.
                # train_metric(y_batch_train, logits)

            # # Display metrics at the end of each epoch.
            # train_acc = train_metric.result()
            # print('Training acc over epoch: %s' % (float(train_acc),))
            # # Reset training metrics at the end of each epoch
            # train_metric.reset_states()
            #
            # # Run a validation loop at the end of each epoch.
            # for x_batch_val, y_batch_val in val_data:
            #     val_logits = model(x_batch_val)
            #     # Update val metrics
            #     val_metric(y_batch_val, val_logits)
            # val_acc = val_metric.result()
            # val_metric.reset_states()
            # print('Validation acc: %s' % (float(val_acc),))

            # 保存模型
            self._save_model(model, epoch, step=-1)

    # 利用测试集评估模型
    def evaluate(self, epoch):
        label = keras.Input((*self._label_shape,))
        train_data, val_data, test_data = self.select_datasets()
        model = self.build_model()
        optimizer = self.select_optimier()
        loss = self.select_loss()
        metric = self.select_metric()

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[metric],
                      target_tensors=label
                      )

        # 加载模型参数
        self._load_model(model, epoch=epoch)
        model.evaluate(test_data)

    # 模型预测
    def predict(self):
        pass

    # ==================================动态显示评估模型的一些参考==================================
    # 展示模型结构
    def show_model(self):
        model = self.build_model()
        model.summary()
        keras.utils.plot_model(model, self._model_name + '_model.png')
        keras.utils.plot_model(model, self._model_name + '_info.png', show_shapes=True)

    # 展示模型训练时的梯度变化（训练时，默认动态展示）
    def show_model_grads_on_train(self):
        pass

    # 展示模型训练时你所关心的评估的变化（训练时，默认动态展示）
    def show_model_metric_on_train(self):
        pass

    # ==================================构建训练模型和测试模型的一些组件=================================
    # 加载模型
    def _load_model(self, model, epoch=0):
        if epoch==0:
            # 文件名格式:modelname-weight_epoch
            for filename in os.listdir(self._save_model_dir):
                filename_list = filename.split('.')[0].split('_')
                if(filename_list[0]==self._model_name + '-weight'):
                    epoch = max(int(filename_list[1]), epoch)
        if epoch>0:
            filename = self._model_name + '-weight_' + str(epoch)
            save_root = os.path.join(self._save_model_dir, filename)
            model.load_weights(save_root)
            print('第{}epoch模型加载成功'.format(epoch))
        return epoch

    # 保存模型
    def _save_model(self, model, epoch, step=0):
        # 文件名格式:chinesestring-crnn-weight_epoch_step
        filename = self._model_name + '-weight_' + str(epoch) + str(step)
        save_root = os.path.join(self._save_model_dir, filename)
        model.save_weights(save_root)
        print('第{}epoch第{}step模型保存成功'.format(epoch, step))

    # 准备数据（训练集、验证集）
    def select_datasets(self):
        train_data , val_data, test_data = self._my_datasets.load_data()
        return train_data, val_data, test_data

    # 选择优化算法
    def select_optimier(self):
        optimier = self._my_optimizer

        return optimier

    # 选着损失函数
    def select_loss(self):
        return self._my_loss

    # 选择评估模型的方法
    def select_metric(self):
        return self._my_metric

    def select_callback(self):
        return self._my_callback

if __name__=='__main__':
    fram = my_frame()
    fram.train()
    # fram.evaluate(251)
    # fram.show_model()