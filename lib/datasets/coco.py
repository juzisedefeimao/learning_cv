import json
import os
import pickle
import tqdm
import numpy as np
from collections import defaultdict
import tensorflow as tf

class coco():
    def __init__(self):
        self._data_dir = 'G:/公有集数据/coco/COCO'
        self._cache_dir = os.path.join(self._data_dir, 'annotations', 'cache')
        path_is_exist(self._cache_dir)
        self._annotation_cache = os.path.join(self._cache_dir, 'annotation_cache.pkl')
        self._classes_cache = os.path.join(self._cache_dir, 'classes_cache.pkl')
        if not os.path.exists(self._annotation_cache) or not os.path.exists(self._classes_cache):
            self._get_annotations()
        self._annotations, self._train_file, self._classes = self._read_annotations()

    def load_data(self):
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(self._train_file))
        dataset = dataset.shuffle(2000).map(self._map_func).batch(10)
        return dataset

    def _map_func(self, image_name):
        # image_path = os.path.join(self._data_dir, 'train2017', image_name)
        image_path = image_name
        # annotation = self._annotations[image_name]
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (416, 416))
        return image

    def _read_annotations(self):
        with open(self._annotation_cache, 'rb') as f:
            annotations = pickle.load(f)
        with open(self._classes_cache, 'rb') as f:
            classes = pickle.load(f)

        train_cache = os.path.join(self._cache_dir, 'train_cache.pkl')
        if not os.path.exists(train_cache):
            train_file = [os.path.join(self._data_dir, 'train2017', image_name) for image_name, value in annotations.items()]
            with open(train_cache, 'wb') as f:
                pickle.dump(train_file, f)
        else:
            with open(train_cache, 'rb') as f:
                train_file = pickle.load(f)

        return annotations, train_file, classes

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        '''Preprocess true boxes to training input format

        Parameters
        ----------
        true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), wh
        num_classes: integer

        Returns
        -------
        y_true: list of array, shape like yolo_outputs, xywh are reletive value

        '''
        assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
        num_layers = len(anchors) // 3  # default setting
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                           dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):
            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find best anchor for each true box
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true

    def _get_annotations(self):
        id_name = {}
        with open("G:/公有集数据/coco/COCO/annotations/instances_train2017.json",
            encoding='utf-8') as f:
            data = json.load(f)
        categories = data['categories']
        classes_name = {}
        id_classes = {}
        for i, name in enumerate(categories):
            classes_name[i] = name['name']
            id_classes[name['id']] = i
        annotations = data['annotations']
        for annotation in tqdm.tqdm(annotations):

            image_name = '{0:0>12}.jpg'.format(annotation['image_id'])
            category_id = id_classes[annotation['category_id']]

            if image_name not in id_name:
                id_name[image_name] = []
            id_name[image_name].append([annotation['bbox'], category_id])

        with open(self._annotation_cache, 'wb') as f:
            pickle.dump(id_name, f)
        with open(self._classes_cache, 'wb') as f:
            pickle.dump(classes_name, f)

def path_is_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__=='__main__':
    co = coco()
    data = co.load_data()
    for da in data:
        print(da)
    # co._read_annotations()
    # co.load_data()
    # a ='{0:0>5}'.format(32)
    # print(a)