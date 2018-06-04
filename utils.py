import argparse

from keras import optimizers
from keras.callbacks import Callback

# from vgg19 import VGG19
# from inception_v3 import InceptionV3
# from xception import Xception
from resnet50 import ResNet50
from vgg16 import VGG16

NUM_CLASS = 121
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNEL = 3


def get_nets(net_name):
  nets_dict = {
    'vgg16': VGG16,
    # 'vgg19': VGG19,
    # 'inceptionv3': InceptionV3,
    # 'xception': Xception,
    'resnet50': ResNet50
  }
  nets = nets_dict[net_name]
  return nets


def get_optimizer(name, l_r, decay=0.):
  if name == 'sgd':
    optimizer = optimizers.SGD(lr=l_r, momentum=0., decay=decay, nesterov=False)
  elif name == 'rmsprop':
    optimizer = optimizers.RMSprop(lr=l_r, rho=0.9, decay=decay)
  elif name == 'adagrad':
    optimizer = optimizers.Adagrad(lr=l_r, decay=decay)
  elif name == 'adadelta':
    optimizer = optimizers.Adadelta(lr=l_r, rho=0.95, decay=decay)
  elif name == 'adam':
    optimizer = optimizers.Adam(lr=l_r, beta_1=0.9, beta_2=0.999, decay=0.1 * decay)
  elif name == 'adamax':
    optimizer = optimizers.Adamax(lr=l_r, beta_1=0.9, beta_2=0.999, decay=0.1 * decay)
  elif name == 'nadam':
    optimizer = optimizers.Nadam(lr=l_r, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
  else:
    optimizer = None
  return optimizer


def normalize(data_flow):
  for image, label in data_flow:
    image = image / 127.5 - 1
    yield image, label


def str2bool(inputs):
  if inputs.lower() in ('yes', 'true', 'y', 't', '1'):
    return True
  elif inputs.lower() in ('no', 'false', 'n', 'f', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')
