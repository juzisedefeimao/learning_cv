import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import losses_utils
from tensorflow.tools.docs import doc_controls
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.framework import sparse_tensor
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
import numpy as np
import abc
import six
K = keras.backend

class Loss(object):
  """Loss base class.

  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

  Example subclass implementation:
  ```
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = ops.convert_to_tensor(y_pred)
      y_true = math_ops.cast(y_true, y_pred.dtype)
      return K.mean(math_ops.square(y_pred - y_true), axis=-1)
  ```

  Args:
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    self.reduction = reduction
    self.name = name

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional `Tensor` whose rank is either 0, or the same rank
        as `y_true`, or is broadcastable to `y_true`. `sample_weight` acts as a
        coefficient for the loss. If a scalar is provided, then the loss is
        simply scaled by the given value. If `sample_weight` is a tensor of size
        `[batch_size]`, then the total loss for each sample of the batch is
        rescaled by the corresponding element in the `sample_weight` vector. If
        the shape of `sample_weight` matches the shape of `y_pred`, then the
        loss of each measurable element of `y_pred` is scaled by the
        corresponding value of `sample_weight`.

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
        shape as `y_true`; otherwise, it is scalar.

    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    scope_name = 'lambda' if self.name == '<lambda>' else self.name
    with ops.name_scope(scope_name, format(self.__class__.__name__),
                        (y_pred, y_true, sample_weight)):
      losses = self.call(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self.reduction)

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Loss` instance.
    """
    return cls(**config)

  def get_config(self):
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values, with the same shape as 'y_pred'.
      y_pred: The predicted values.
    """
    NotImplementedError('Must be implemented in subclasses.')


class LossFunctionWrapper(Loss):
  """Wraps a loss function in the `Loss` class.

  Args:
    fn: The loss function to wrap, with signature `fn(y_true, y_pred,
      **kwargs)`.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: (Optional) name for the loss.
    **kwargs: The keyword arguments that are passed on to `fn`.
  """

  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None,
               **kwargs):
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    return self.fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class SparseCategoricalCrossentropy(LossFunctionWrapper):


  def __init__(self,
               from_logits=False,
               label_smooth_c=0.1,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(SparseCategoricalCrossentropy, self).__init__(
        categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smooth_c=label_smooth_c)

def categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1, label_smooth_c=0.1):
    if label_smooth_c==0:
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)
    else:
        return label_smooth_categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis,
                                              label_smooth_c=label_smooth_c)

def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)

def label_smooth_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1, label_smooth_c=0.1):

    shape = y_pred.get_shape()
    y_true = tf.cast(y_true, dtype=tf.int64)
    y_true = tf.one_hot(y_true, shape[-1])
    y_true = tf.cast(y_true, dtype=tf.float32) * (1-label_smooth_c-(label_smooth_c/(shape[-1]-1)))
    y_true = y_true + tf.ones_like(y_true)*(label_smooth_c/(shape[-1]-1))

    return K.categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)

class CTC_Loss(LossFunctionWrapper):


  def __init__(self,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(CTC_Loss, self).__init__(
        ctc_loss,
        name=name,
        reduction=reduction)

def ctc_loss(y_true, y_pred):
    return K.ctc_batch_cost(y_true, y_pred, input_length=np.ones((32,1))*70, label_length=np.ones((32,1))*10)

class YOLO_Loss(LossFunctionWrapper):


  def __init__(self,
               reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(YOLO_Loss, self).__init__(
        yolo_loss,
        name=name,
        reduction=reduction)

def yolo_loss(y_true, y_pred, S=7, B=2, classes=20):
    """
    :param y_true: [S,S,B*5+classes] [x,y,h,w,obj, x,y,h,w,obj,,,]
    :param y_pred: [S,S,B*5+classes]
    :return: float
    """
    y_true_shape = y_true.shape
    assert y_true_shape[-1] == B * 5 + classes, \
        '输出维度{}不等于B*5+classes：{}'.format(y_true_shape[-1], B * 5 + classes)
    y_pred_shape = y_pred.shape
    assert y_pred_shape[-1] == B * 5 + classes, \
        '输出维度{}不等于B*5+classes：{}'.format(y_pred_shape[-1], B * 5 + classes)

    return 3.0