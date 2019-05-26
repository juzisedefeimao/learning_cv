from cv_learning.lib.model_build.mymodels.crnn import crnn_net
from cv_learning.lib.model_build.mymodels.darknet import darknet53_depthwise, darknet53, darknet53_depthwisev2
from cv_learning.lib.model_build.mymodels.yolo import yolo

class models_factory():
    def __init__(self, netconfig):
        self._config = netconfig

    def load_net(self):
        my_net = self._get_net(self._config)
        if 'input_shape' in self._config:
            input_shape = self._config['input_shape']
        else:
            raise KeyError('未设置网络{}的输入张量形状'.format(self._config['name']))
        if 'label_shape' in self._config:
            label_shape = self._config['label_shape']
        else:
            raise KeyError('未设置网络{}的标签形状'.format(self._config['name']))
        return my_net, input_shape, label_shape

    def _get_net(self, netconfig):
        netname = netconfig['name']
        print('网络模型:使用{}'.format(netname))
        if netname == 'crnn':
            return self._crnn_init(netconfig)
        if netname == 'darknet53':
            return self._darknet53_init(netconfig)
        if netname == 'darknet53v2':
            return self._darknet53v2_init(netconfig)
        if netname == 'crnn':
            return self._crnn_init(netconfig)
        if netname == 'yolo':
            return self._yolo_init(netconfig)
        else:
            raise KeyError('没有选择的网络{}'.format(netname))

    def _crnn_init(self, netconfig):
        return crnn_net()

    def _darknet53_init(self, netconfig):
        if netconfig['use_depthwise']:
            darknet = darknet53_depthwise
        else:
            darknet = darknet53
        return darknet(include_top=netconfig['include_top'],
                        pooling=netconfig['pooling'],
                       classes = netconfig['classes'])

    def _darknet53v2_init(self, netconfig):
        darknet = darknet53_depthwisev2
        return darknet(include_top=netconfig['include_top'],
                       pooling=netconfig['pooling'],
                       classes=netconfig['classes'])

    def _yolo_init(self, netconfig):
        return yolo(S = netconfig['S'],
                    B = netconfig['B'],
                    classes=netconfig['classes'])