from cv_learning.lib.model_frame.mymodel_frame import my_frame

class crnn_model(my_frame):
    def __init__(self):
        super(crnn_model, self).__init__(yaml_dir='D:\\jjj\\GitHub\\cv_learning\\model_factory\\OCR\\train\\config.yaml')


if __name__=='__main__':
    cr = crnn_model()
    # cr.show_model()
    cr.train()