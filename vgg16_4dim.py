# -*- coding: utf-8 -*-
"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import, print_function

import warnings

from keras import backend as K
#from keras.utils.data_utils import get_file
from keras import regularizers
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Concatenate
from keras.layers import Conv2D as Conv2DOrg
from keras.layers import Dense as DenseOrg
from keras.layers import (Dropout, Flatten, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Input, MaxPooling2D)
from keras.models import Model
from keras.utils import layer_utils

#from utils import decode_predictions
#from utils import preprocess_input
from utils import _obtain_input_shape

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


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


def VGG16(include_top=True,
          weights='imagenet',
          input_tensor1=None,
          input_tensor2=None,
          input_shape1=None,
          input_shape2=None,
          pooling=None,
          classes=1000,
          drop_rate=0.5):
  """Instantiates the VGG16 architecture.

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
      include_top: whether to include the 3 fully-connected
          layers at the top of the network.
      weights: one of `None` (random initialization)
          or "imagenet" (pre-training on ImageNet).
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 48.
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
  # input_shape1 = _obtain_input_shape(input_shape1,
  #                                   default_size=224,
  #                                   min_size=48,
  #                                   data_format=K.image_data_format(),
  #                                   include_top=include_top)
  # input_shape2 = _obtain_input_shape(input_shape2,
  #                                   default_size=56,
  #                                   min_size=48,
  #                                   data_format=K.image_data_format(),
  #                                   include_top=include_top)

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

  x = Concatenate()([img_input, hm_input])

  # Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_alt')(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
  # model.layers[:4]

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
  # model.layers[:7]

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
  # model.layers[:11]

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
  # model.layers[:15]

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
  # model.layers[:19]

  if include_top:
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    if drop_rate is not None:
      x = Dropout(drop_rate)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    if drop_rate is not None:
      x = Dropout(drop_rate)(x)
    #x = Dense(classes, activation='softmax', name='predictions')(x)
    x = Dense(classes, activation='softmax', name='prediction')(x)
    # model.layers[:23]
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
  model = Model([input1, input2], x, name='vgg16')

  return model
