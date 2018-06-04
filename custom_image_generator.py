"""
For input rgb img with attention map, we can't use keras's ImageDataGenerator to do augmentation.
Most code here is basically copied from keras.preprocessing.image.
"""

import os
import numpy as np
from numpy import concatenate
import warnings
import multiprocessing
from functools import partial

from PIL import ImageEnhance
from PIL import Image as pil_image

from keras.preprocessing.image import load_img, img_to_array, flip_axis, apply_transform, transform_matrix_offset_center, random_brightness, _iter_valid_files, _count_valid_files_in_directory, _list_valid_filenames_in_directory

def read_image_from_folder(directory):
    """
    Return   (sample count, file names, classes(in shape(file names,)), class indicies(map label to index))
    """
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

    # first, count the number of samples and classes
    samples = 0

    label_names = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            label_names.append(subdir)
    num_classes = len(label_names)
    class_indices = dict(zip(label_names, range(len(label_names))))

    pool = multiprocessing.pool.ThreadPool()
    function_partial = partial(_count_valid_files_in_directory,
                                white_list_formats=white_list_formats,
                                follow_links=False,
                                split=None)
    samples = sum(pool.map(function_partial,
                                (os.path.join(directory, subdir)
                                    for subdir in label_names)))

    print('Found %d images belonging to %d classes.' % (samples, num_classes))

    # second, build an index of the images in the different class subfolders
    results = []

    filenames = []
    
    i = 0
    for dirpath in (os.path.join(directory, subdir) for subdir in label_names):
        results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                        (dirpath, white_list_formats, None,
                                            class_indices, False)))

    classes = np.zeros((samples,), dtype='int32')
    for res in results:
        tclasses, tfilenames = res.get()
        classes[i:i + len(tclasses)] = tclasses
        filenames += tfilenames
        i += len(tclasses)

    pool.close()
    pool.join()

    return (samples, filenames, classes, class_indices)

def random_transform( 
    x, 
    attentionmap,
    rotation_range, 
    vertical_flip, 
    horizontal_flip, 
    height_shift_range, 
    width_shift_range,
    shear_range,
    brightness_range,
    seed=None):
    """Randomly augment a single image tensor.

    # Arguments
        x: 3D tensor, single image.
        seed: random seed.

    # Returns
        A randomly transformed version of the input (same shape).
    """
    
    # use composition of homographies
    # to generate final transform that needs to be applied
    if rotation_range:
        theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
    else:
        theta = 0

    if height_shift_range:
        try:  # 1-D array-like or int
            tx = np.random.choice(height_shift_range)
            tx *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            tx = np.random.uniform(-height_shift_range,
                                    height_shift_range)
        if np.max(height_shift_range) < 1:
            tx *= x.shape[0]
    else:
        tx = 0

    if width_shift_range:
        try:  # 1-D array-like or int
            ty = np.random.choice(width_shift_range)
            ty *= np.random.choice([-1, 1])
        except ValueError:  # floating point
            ty = np.random.uniform(-width_shift_range,
                                    width_shift_range)
        if np.max(width_shift_range) < 1:
            ty *= x.shape[1]
    else:
        ty = 0

    if shear_range:
        shear = np.deg2rad(np.random.uniform(-shear_range, shear_range))
    else:
        shear = 0

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                    [0, 1, ty],
                                    [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                [0, np.cos(shear), 0],
                                [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)


    if transform_matrix is not None:
        h, w = x.shape[0], x.shape[1]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, 2,
                            fill_mode='nearest')
        attentionmap = apply_transform(attentionmap, transform_matrix, 2,
                            fill_mode='nearest')
        
    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, 1)
            attentionmap = flip_axis(attentionmap, 1)

    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, 0)
            attentionmap = flip_axis(attentionmap, 0)

    if brightness_range is not None:
        x = random_brightness(x, brightness_range)

    return x, attentionmap

class ImageDataGeneratorWithAttention():
    """Iterator specially made for this
    Return (batch, x, attention, label)
    """
    def __init__(self, trainfolder, attentionfolder, 
        rotation_range=5.0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.8, 1.2),
        fill_mode='wrap',
        horizontal_flip=True,
        vertical_flip=True,
        batch_size = 32, 
        interpolation = 'bilinear'
        ):
        self.train_folder = trainfolder
        self.attention_folder = attentionfolder

        (train_count, train_filenames, train_labels, class_indicies) = read_image_from_folder(trainfolder)

        self.num_classes = len(class_indicies)
        self.img_count = train_count
        self.train_filenames = train_filenames
        self.train_labels = train_labels
        self.index_array = np.random.permutation(self.img_count)
        
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.fill_mode = fill_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.batch_size = batch_size
        self.interpolation = interpolation

    def flow(self):
        current_index = 0
        while True:
        
            start_index = current_index
            end_index = start_index + self.batch_size
            end_index = min(end_index, self.img_count)
            count = end_index - start_index
            current_index = end_index

            batch_x = np.zeros((count, 224, 224, 3,))
            batch_a = np.zeros((count, 224, 224, 1,))

            if count == 0:
                self.index_array = np.random.permutation(self.img_count)
                current_index = 0
                continue

            file_indexes = self.index_array[start_index : count]
            for i, j in enumerate(file_indexes):

                from PIL import ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True

                img = load_img(os.path.join(self.train_folder, self.train_filenames[j]),
                    grayscale=False,
                    target_size=(224,224),
                    interpolation=self.interpolation)
                
                attention_file_name = os.path.splitext(self.train_filenames[j])[0]
                
                attention_img = load_img(os.path.join(self.attention_folder, attention_file_name + '.png'),
                    grayscale=True,
                    target_size=(224,224),
                    interpolation=self.interpolation)
            
                x = img_to_array(img)
                attention = img_to_array(attention_img)
                x, attention = random_transform(x, attention, 
                    self.rotation_range, 
                    self.vertical_flip, 
                    self.horizontal_flip, 
                    self.height_shift_range, 
                    self.width_shift_range, 
                    0.0, 
                    self.brightness_range)

                x = (x - 127.5) / 127.5

                #attention normalization, Copied from grocery code of Jiangke Lin
                def scale255(inputs):
                    input_max = np.max(inputs)
                    if input_max == 0:
                        input_max = 2
                    outputs = inputs * float(255.0 / input_max)
                    return outputs
                attention = scale255(attention) - 127.5

                batch_x[i] = x
                batch_a[i] = attention

            batch_y = np.zeros((len(batch_x), self.num_classes))
            for i, label in enumerate(self.train_labels[np.array(file_indexes)]):
                batch_y[i, label] = 1.
            yield [batch_x, batch_a], batch_y

"""
#Test if it works.
test = ImageDataGeneratorWithAttention(
    './data/train',
    './data/attention'
)
import matplotlib.pyplot as plt

for ([batch_x, batch_a], batch_y) in test.flow():
    for t in range(batch_x.shape[0]):
        print(batch_y[t])
        plt.imshow(batch_x[t,...].astype(np.uint8))
        plt.show()
"""