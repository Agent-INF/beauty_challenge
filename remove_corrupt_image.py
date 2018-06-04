import argparse
import os

from PIL import Image


def main(args):
  data_dir = os.path.join('data', args.dataname)
  dir_list = os.listdir(data_dir)
  for file_dir in dir_list:
    for file_path in os.listdir(os.path.join(data_dir, file_dir)):
      full_path = os.path.join(data_dir, file_dir, file_path)
      try:
        _ = Image.open(full_path)
      except IOError:
        os.remove(full_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataname', type=str, help='dataset name')
  ARGS = parser.parse_args()
  main(ARGS)
