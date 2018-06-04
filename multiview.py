import argparse
import os

import tensorflow as tf
from keras.backend import int_shape
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (Add, AveragePooling2D, Concatenate, Dense, Flatten,
                          Input, MaxPooling2D)
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file

from resnet50 import conv_block, identity_block
from resnet50_4dim import ResNet50 as ResNet50_4dim
from utils import (IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH, get_nets, get_optimizer,
                   normalize, str2bool)

from custom_image_generator import ImageDataGeneratorWithAttention

URL_PREFIX = 'https://github.com/fchollet/deep-learning-models/releases/download/'
MODEL_VERSION = {
  'vgg16': 'v0.1/',
  'vgg19': 'v0.1/',
  'inceptionv3': 'v0.5/',
  'xception': 'v0.4/',
  'resnet50': 'v0.2/'
}
MODEL_NAME = {
  'vgg16': 'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
  'vgg19': 'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
  'inceptionv3': 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
  'xception': 'xception_weights_tf_dim_ordering_tf_kernels.h5',
  'resnet50': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
}


class Network(object):

  def __init__(self, args):
    self._mode = args.mode
    self._batch_size = args.batch_size
    self._initial_epoch = args.initial_epoch
    self._epochs = args.epochs
    self._dataname = args.dataname
    self._suffix = args.suffix
    self._optimizer = get_optimizer(args.optimizer, args.lr)
    self._data_dir = os.path.join('data', self._dataname)
    self._data_attention_folder = os.path.join('data', self._dataname + '_attention')
    self._ckpt_dir = os.path.join('checkpoint', self._dataname, self._suffix)
    self._sample_dir = os.path.join('sample', self._dataname, self._suffix)
    self._log_dir = os.path.join('log', self._dataname, self._suffix)
    self._nets_name = args.nets
    self._nets = get_nets(args.nets)
    self._num_class = len(os.listdir(self._data_dir))
    self._train_view_model = args.train_view_model
    self._print_layer_name = args.print_layer_name

    self._full_weight_path = os.path.join(
      'checkpoint', args.full_weights_path) if args.full_weights_path else None
    self._view1_weight_path = os.path.join(
      'checkpoint', args.view1_weights_path) if args.view1_weights_path else None
    self._view2_weight_path = os.path.join(
      'checkpoint', args.view2_weights_path) if args.view2_weights_path else None
    self._view3_weight_path = os.path.join(
      'checkpoint', args.view3_weights_path) if args.view3_weights_path else None

    for directory in [self._data_dir, self._ckpt_dir, self._sample_dir, self._log_dir]:
      if not os.path.isdir(directory):
        os.makedirs(directory)

  def _resnet50_stage5_model(self, name, input_shape=None):  # , input_tensor=None):
    inputs = Input(shape=input_shape)
    if input_shape[-1] == 1024:
      outputs = conv_block(inputs, 3, [512, 512, 2048], stage=5, block='a')
    else:
      outputs = conv_block(inputs, 3, [512, 512, 2048], stage=5, block='a1')
    outputs = identity_block(outputs, 3, [512, 512, 2048], stage=5, block='b')
    outputs = identity_block(outputs, 3, [512, 512, 2048], stage=5, block='c')
    outputs = AveragePooling2D((7, 7), name='max_pool')(outputs)
    # outputs = MaxPooling2D((7, 7), name='max_pool')(outputs)
    outputs = Flatten()(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)

  def setup_model(self):
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL)
    attention_shape = (IMG_WIDTH, IMG_HEIGHT, 1)
    self.image_input = Input(shape=input_shape)
    self.attention_input = Input(shape=attention_shape)
    # TODO: change following network name, eg. change view to others
    view_names = ['view1', 'view2', 'view3']
    with tf.device('/gpu:0'):
      self.view1_model = self._nets(
        include_top=True,
        input_shape=input_shape,
        get_feature_stage=4,
        name=view_names[0])
      if not self._full_weight_path:
        self._load_weights(self.view1_model, self._view1_weight_path, name=view_names[0])
      self._rename_model_layers(self.view1_model, view_names[0])
      view1 = self.view1_model(self.image_input)

      self.view2_model = self._nets(
        include_top=True,
        input_shape=input_shape,
        get_feature_stage=4,
        name=view_names[1])
      if not self._full_weight_path:
        self._load_weights(self.view2_model, self._view2_weight_path, name=view_names[1])
      self._rename_model_layers(self.view2_model, view_names[1])
      view2 = self.view2_model(self.image_input)

    with tf.device('/gpu:1'):
      self.view3_model = ResNet50_4dim(
        include_top=True,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL + 1),
        get_feature_stage=4,
        name=view_names[2])
      if not self._full_weight_path:
        self._load_weights(self.view3_model, self._view3_weight_path, name=view_names[2])
      self._rename_model_layers(self.view3_model, view_names[2])
      img_attention_concat = Concatenate(axis = -1, name = 'img_attention_concat')([self.image_input, self.attention_input])
      view3 = self.view3_model(img_attention_concat)

      view_concat = Concatenate(axis=-1, name='view_concat')([view1, view2, view3])

      self.pre_aggregation_model = self._resnet50_stage5_model(
        name='pre_aggregation', input_shape=int_shape(view_concat)[1:])
      pre_aggregation = self.pre_aggregation_model(view_concat)
      if not self._full_weight_path:
        self._load_weights(self.pre_aggregation_model, None, 'pre_aggregation')
      self._rename_model_layers(self.pre_aggregation_model, 'pre_aggregation')
      pre_aggregation = Dense(
        self._num_class, activation='softmax', name='pre_prediction')(
          pre_aggregation)

      self.shared_model = self._resnet50_stage5_model(
        name='shared', input_shape=int_shape(view1)[1:])
      if not self._full_weight_path:
        self._load_weights(self.shared_model, None, 'shared')
      self._rename_model_layers(self.shared_model, 'shared')

      view1_feature = self.shared_model(view1)
      view2_feature = self.shared_model(view2)
      view3_feature = self.shared_model(view3)
      feat_concat = Concatenate(axis=-1)([view1_feature, view2_feature, view3_feature])
      post_aggregation = Dense(
        self._num_class, activation='softmax', name='post_prediction')(
          feat_concat)

      final_score = Add(name='predict_fusion')([pre_aggregation, post_aggregation])
      self.model = Model(inputs=[self.image_input, self.attention_input], outputs=final_score, name='model')

      view_models = [self.view1_model, self.view2_model, self.view3_model]

    try:  #loading full model weights may fail due to same layers trainable status has been changed
      if not self._train_view_model:
        self._change_layer_trainable_status(view_models, False)
      if self._full_weight_path:
        self._load_weights(self.model, self._full_weight_path, 'full')
    except ValueError:
      print('Loading full model weights failed, trying to fix this issue...')
      if self._train_view_model:
        self._change_layer_trainable_status(view_models, False)
        if self._full_weight_path:
          self._load_weights(self.model, self._full_weight_path, 'full')
        self._change_layer_trainable_status(view_models, True)
      else:
        self._change_layer_trainable_status(view_models, True)
        if self._full_weight_path:
          self._load_weights(self.model, self._full_weight_path, 'full')
        self._change_layer_trainable_status(view_models, False)
      print('Loading full model weights success!')

    # self.vis_name = [x for x in self.model.layers]
    # self.model.summary()

  def _change_layer_trainable_status(self, models, status):
    if not isinstance(models, (tuple, list)):
      models = [models]
    for model in models:
      for layer in model.layers:
        layer.trainable = status

  def _print_model_layer_name(self, model):
    if self._print_layer_name:
      for i, layer in enumerate(model.layers):
        print(i, layer.name)

  def _load_weights(self, model, path, name):
    if path is None:
      path = get_file(
        MODEL_NAME[self._nets_name],
        URL_PREFIX + MODEL_VERSION[self._nets_name] + MODEL_NAME[self._nets_name],
        cache_subdir='models')
      print('Loading Keras Pretrained Model for ' + name)
    else:
      print('Loading Local Pretrained Model for ' + name)
      print('From ' + path)
    # self._print_model_layer_name(model)
    model.load_weights(path, by_name=True)  # , skip_mismatch=True)

  def _rename_model_layers(self, model, prefix):
    for layer in model.layers:
      layer.name = prefix + '_' + layer.name
    self._print_model_layer_name(model)

  def train(self):
    self.setup_model()
    self.model.compile(
      optimizer=self._optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    """
    train_datagen = ImageDataGenerator(
      # featurewise_center=True,
      # featurewise_std_normalization=True,
      # samplewise_center=True,
      # samplewise_std_normalization=True,
      rotation_range=5.0,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=(0.8, 1.2),
      fill_mode='wrap',
      horizontal_flip=True,
      vertical_flip=True,
      validation_split=0.1)
    train_generator = normalize(
      train_datagen.flow_from_directory(
        self._data_dir,
        target_size=(224, 224),
        batch_size=self._batch_size,
        # save_to_dir=self._sample_dir,
        # save_prefix='train',
        # save_format='jpeg',
        subset='training',
        interpolation='bilinear'))
    validate_generator = normalize(
      train_datagen.flow_from_directory(
        self._data_dir,
        target_size=(224, 224),
        batch_size=self._batch_size,
        # save_to_dir=self._sample_dir,
        # save_prefix='validation',
        # save_format='jpeg',
        subset='validation',
        interpolation='bilinear'))
    """
    train_datagen = ImageDataGeneratorWithAttention(
      rotation_range=5.0,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=(0.8, 1.2),
      fill_mode='wrap',
      horizontal_flip=True,
      vertical_flip=True,
      validation_split=0.1
    )

    train_generator = train_datagen.flow(
        self._data_dir, 
        self._data_attention_folder,
        target_size=(224,224),
        batch_size=self._batch_size,
        subset='training', 
        interpolation='bilinear')

    validate_generator = train_datagen.flow(
        self._data_dir, 
        self._data_attention_folder,
        target_size=(224,224),
        batch_size=self._batch_size,
        subset='validation', 
        interpolation='bilinear')

    img_count = sum(
      len(y) for y in
      [os.listdir(os.path.join(self._data_dir, x)) for x in os.listdir(self._data_dir)])
    print('Total Image Numbers: ' + str(img_count))
    num_batches = img_count / self._batch_size
    self.model.fit_generator(
      train_generator,
      steps_per_epoch=int(num_batches / 5 * 4),
      epochs=self._epochs,
      verbose=1,
      callbacks=[
        ModelCheckpoint(
          os.path.join(self._ckpt_dir, 'epoch{epoch:02d}-{val_acc:.2f}.hdf5'),
          verbose=1,
          save_weights_only=True,
          save_best_only=True)
      ],
      validation_data=validate_generator,
      validation_steps=int(num_batches / 5),
      max_queue_size=64,
      workers=self._batch_size // 2,
      use_multiprocessing=False,
      initial_epoch=self._initial_epoch)


def main(args):
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
  config = tf.ConfigProto(allow_soft_placement=True)
  # pylint: disable=E1101
  config.gpu_options.allow_growth = True
  set_session(tf.Session(config=config))
  network = Network(args)
  if args.mode == 'train':
    network.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-m',
    '--mode',
    type=str,
    choices=['train', 'test'],
    default='train',
    help='train or test mode.')
  parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
  parser.add_argument('-g', '--gpuid', type=str, default=None, help='use which gpus')
  parser.add_argument('-d', '--dataname', type=str, default=None, help='dataset name')
  parser.add_argument('-i', '--initial_epoch', type=int, default=0, help='initial epoch')
  parser.add_argument(
    '-e',
    '--epochs',
    type=int,
    default=100,
    help='final epochs, training epochs is final - initial')
  parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate')
  parser.add_argument(
    '-n',
    '--nets',
    type=str,
    choices=['vgg16', 'vgg19', 'inceptionv3', 'xception', 'resnet50'],
    default='resnet50',
    help='network type')
  parser.add_argument('-s', '--suffix', type=str, default='', help='saving models suffix')
  parser.add_argument(
    '-o',
    '--optimizer',
    type=str,
    choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
    default='adam',
    help='optimizer type')
  parser.add_argument(
    '-t',
    '--train_view_model',
    type=str2bool,
    default=False,
    help='whether to train view model parameters.')
  parser.add_argument(
    '-p',
    '--print_layer_name',
    type=str2bool,
    default=False,
    help='whether to print all model layers name')
  parser.add_argument(
    '-w', '--full_weights_path', type=str, default=None, help='custom model weights path')
  parser.add_argument(
    '-x', '--view1_weights_path', type=str, default=None, help='view1 model weights path')
  parser.add_argument(
    '-y', '--view2_weights_path', type=str, default=None, help='view2 model weights path')
  parser.add_argument(
    '-z', '--view3_weights_path', type=str, default=None, help='view3 model weights path')
  ARGS = parser.parse_args()
  print(ARGS)
  main(ARGS)
