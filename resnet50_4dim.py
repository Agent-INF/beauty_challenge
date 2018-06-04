# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import, print_function

import warnings

from keras import backend as K
from keras import layers, regularizers
#from keras.utils.data_utils import get_file
#from utils import decode_predictions
#from utils import preprocess_input
# from utils import _obtain_input_shape
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Concatenate)
from keras.layers import Conv2D as Conv2DOrg
from keras.layers import Dense as DenseOrg
from keras.layers import (Dropout, Flatten, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Input, MaxPooling2D,
                          ZeroPadding2D)
from keras.models import Model
from keras.utils import layer_utils

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Conv2D(filters,
           kernel_size,
           strides=(1, 1),
           padding='valid',
           data_format=None,
           dilation_rate=(1, 1),
           activation=None,
           use_bias=True,
           kernel_initializer='glorot_uniform',
           bias_initializer='zeros',
           kernel_regularizer=regularizers.l2(1e-5),
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           **kwargs):
  return Conv2DOrg(
    filters,
    kernel_size,
    strides=strides,
    padding=padding,
    data_format=data_format,
    dilation_rate=dilation_rate,
    activation=activation,
    use_bias=use_bias,
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer,
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
    kernel_constraint=kernel_constraint,
    bias_constraint=bias_constraint,
    **kwargs)


def Dense(units,
          activation=None,
          use_bias=True,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros',
          kernel_regularizer=regularizers.l2(1e-5),
          bias_regularizer=None,
          activity_regularizer=None,
          kernel_constraint=None,
          bias_constraint=None,
          **kwargs):
  return DenseOrg(
    units,
    activation=activation,
    use_bias=use_bias,
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer,
    kernel_regularizer=kernel_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
    kernel_constraint=kernel_constraint,
    bias_constraint=bias_constraint,
    **kwargs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  # Arguments
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filterss of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  # Returns
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  x = layers.add([x, input_tensor])
  x = Activation('relu')(x)
  return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
  """conv_block is the block that has a conv layer at shortcut

  # Arguments
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filterss of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  # Returns
      Output tensor for the block.

  Note that from stage 3, the first conv layer at main path is with strides=(2,2)
  And the shortcut should have strides=(2,2) as well
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  shortcut = Conv2D(
    filters3, (1, 1), strides=strides, name=conv_name_base + '1')(
      input_tensor)
  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])
  x = Activation('relu')(x)
  return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor1=None,
             input_tensor2=None,
             input_shape1=None,
             input_shape2=None,
             pooling=None,
             classes=1000,
             drop_rate=0.5):
  """Instantiates the ResNet50 architecture.

  Optionally loads weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_data_format="channels_last"` in your Keras config
  at ~/.keras/keras.json.

  The model and the weights are compatible with both
  TensorFlow and Theano. The data format
  convention used by the model is the one
  specified in your Keras config file.

  # Arguments
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization)
          or "imagenet" (pre-training on ImageNet).
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 197.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.

  # Returns
      A Keras model instance.

  # Raises
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  if weights not in {'imagenet', None}:
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization) or `imagenet` '
                     '(pre-training on ImageNet).')

  # if weights == 'imagenet' and include_top and classes != 1000:
  #    raise ValueError('If using `weights` as imagenet with `include_top`'
  #                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  if input_tensor1 is None:
    img_input = Input(shape=input_shape1)
  else:
    if not K.is_keras_tensor(input_tensor1):
      img_input = Input(tensor=input_tensor1, shape=input_shape1)
    else:
      img_input = input_tensor1
  if input_tensor2 is None:
    hm_input = Input(shape=input_shape2)
  else:
    if not K.is_keras_tensor(input_tensor2):
      hm_input = Input(tensor=input_tensor2, shape=input_shape2)
    else:
      hm_input = input_tensor2

  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1

  x = Concatenate()([img_input, hm_input])

  x = ZeroPadding2D((3, 3))(x)
  x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1_alt')(x)
  x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)
  # model.layers[:6]

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
  # model.layers[:38]

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
  # model.layers[:80]

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
  # model.layers[:142]

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = AveragePooling2D((7, 7), name='avg_pool')(x)
  # model.layers[:175]

  if include_top:
    x = Flatten()(x)
    x = Dropout(drop_rate)(x)
    #x = Dense(classes, activation='softmax', name='fc1000')(x)
    x = Dense(classes, activation='softmax', name='prediction')(x)
    # model.layers[:178]
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor1 is not None:
    input1 = get_source_inputs(input_tensor1)
  else:
    input1 = img_input
  if input_tensor2 is not None:
    input2 = get_source_inputs(input_tensor2)
  else:
    input2 = hm_input
  # Create model.
  model = Model([input1, input2], x, name='resnet50')

  return model
