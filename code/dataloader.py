import glob
from collections import defaultdict
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


@tf.function
def augment(image):

    # Shuffle RGB channel
    image = tf.transpose(image, perm=(2, 0, 1))
    image = tf.random.shuffle(image)
    image = tf.transpose(image, perm=(1, 2, 0))

    # jpeg quality
    image = tf.image.random_jpeg_quality(image, 50, 100)

    # Random rotation 90 degrees
    image = tf.image.rot90(image, k=tf.random.uniform(
        minval=0, maxval=4, dtype=tf.int32, shape=[]))

    # Random translation
    image = tf.pad(image, paddings=[[20, 20], [20, 20], [
        0, 0]], mode='CONSTANT', constant_values=0)
    image = tf.image.random_crop(image, size=(100, 100, 3))

    # Random flips
    image = tf.image.random_flip_left_right(image)

    # Random color adjust
    r = tf.random.uniform(shape=[], minval=-0.1,
                          maxval=0.1, dtype=tf.float32)
    image = tf.image.adjust_brightness(image, delta=r)
    image = tf.clip_by_value(image, 0.0, 1.0)

    r = tf.random.uniform(shape=[], minval=0.9,
                          maxval=1.3, dtype=tf.float32)
    image = tf.image.adjust_saturation(image, r)
    image = tf.clip_by_value(image, 0.0, 1.0)

    r = tf.random.uniform(shape=[], minval=0.96,
                          maxval=1, dtype=tf.float32)
    image = tf.image.adjust_hue(image, r)
    image = tf.clip_by_value(image, 0.0, 1.0)

    r = tf.random.uniform(shape=[], minval=1.23,
                          maxval=1.25, dtype=tf.float32)
    image = tf.image.adjust_gamma(image, r)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


@tf.function
def batch_augment(images):
    images = tf.map_fn(augment, images, parallel_iterations=12)
    return images


def mixmatch_train_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False):
    def transform(arguments):
        filename = arguments[0]
        label = arguments[1]
        label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
        label = tf.dtypes.cast(label, dtype='int32')
        label = tf.reshape(label, [1])
        label = tf.one_hot(label, 2, dtype='float32')[0]

        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        image = augment(image)

        return image, label

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder=drop_remainder)

    return dataloader


def train_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False):
    def transform(arguments):
        filename = arguments[0]
        label = arguments[1]
        label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
        label = tf.dtypes.cast(label, dtype='int32')
        label = tf.reshape(label, [1])

        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        image = augment(image)

        return image, label

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder=drop_remainder)

    return dataloader


def dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False):
    def transform(arguments):
        filename = arguments[0]
        label = arguments[1]
        label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
        label = tf.dtypes.cast(label, dtype='int32')
        label = tf.reshape(label, [1])

        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        return image, label

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder)

    return dataloader


def pi_model_train_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False):
    def transform(arguments):
        filename = arguments[0]
        label = arguments[1]
        label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
        label = tf.dtypes.cast(label, dtype='int32')
        label = tf.reshape(label, [1])

        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        image1 = augment(image)
        image2 = augment(image)

        return image1, image2, label

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder=drop_remainder)

    return dataloader


def pi_model_unlabeled_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False):
    def transform(arguments):
        filename = arguments
        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        image1 = augment(image)
        image2 = augment(image)
        return image1, image2

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder)

    return dataloader


def mean_teacher_train_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False):
    def transform(arguments):
        filename = arguments[0]
        label = arguments[1]
        label = tf.strings.to_number(label, out_type=tf.dtypes.float32)
        label = tf.dtypes.cast(label, dtype='int32')
        label = tf.reshape(label, [1])

        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        image1 = augment(image)
        image2 = augment(image)

        return image1, image2, label

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder=drop_remainder)

    return dataloader


def mean_teacher_unlabeled_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False):
    def transform(arguments):
        filename = arguments
        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        image1 = augment(image)
        image2 = augment(image)
        return image1, image2

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder)

    return dataloader


def unlabeled_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False, use_augmentation=False):
    def transform(arguments):
        filename = arguments
        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        if use_augmentation == True:
            image = augment(image)
        image = tf.image.resize(image, (100, 100))
        return image

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder)

    return dataloader


def WGAN_dataloader(filenames, batch_size, shuffle, num_parallel_calls, buffer_size, prefetch, drop_remainder=False, use_augmentation=False):
    def transform(arguments):
        filename = arguments
        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        if use_augmentation == True:
            image = augment(image)
        image = tf.image.resize(image, (128, 128))
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle == True:
        dataloader = dataloader.shuffle(buffer_size=buffer_size)
    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size, drop_remainder)

    return dataloader


def survey_dataloader(filenames, batch_size, num_parallel_calls,  prefetch):
    def transform(arguments):
        filename = arguments
        data = tf.io.read_file(filename)
        image = tf.io.decode_png(data)/255
        image = tf.dtypes.cast(image, dtype='float32')
        image = tf.image.resize(image, (100, 100))
        return filename, image

    dataloader = tf.data.Dataset.from_tensor_slices(filenames)

    dataloader = dataloader.map(
        map_func=transform,
        num_parallel_calls=num_parallel_calls,
    )
    dataloader = dataloader.prefetch(prefetch)
    dataloader = dataloader.batch(batch_size)

    return dataloader


def fetch_lense_filenames(path, dataset_name):
    lenses = glob.glob(os.path.join(
        os.getcwd(), path, dataset_name, 'Lenses', '*'))
    lenses = np.array(lenses)
    lenses = np.stack(
        (lenses, np.ones(len(lenses), dtype=np.int32)), axis=-1)
    return lenses


def fetch_nonlense_filenames(path, dataset_name):
    nonlenses = glob.glob(os.path.join(
        os.getcwd(), path, dataset_name, 'NonLenses', '*'))
    nonlenses = np.array(nonlenses)
    nonlenses = np.stack((nonlenses, np.zeros(
        len(nonlenses), dtype=np.int32)), axis=-1)

    return nonlenses


def fetch_unlabeled_filenames(path, dataset_name):
    filenames = glob.glob(os.path.join(
        os.getcwd(), path, dataset_name, '*'))
    filenames = np.array(filenames)

    return filenames


def fetch_survey_filenames(path):
    filenames = glob.glob(os.path.join(path, '*'))
    filenames = np.array(filenames)

    return filenames


def fetch_filenames(path, dataset_name):
    nonlenses = glob.glob(os.path.join(
        os.getcwd(), path, dataset_name, 'NonLenses', '*'))
    nonlenses = np.array(nonlenses)
    nonlenses = np.stack((nonlenses, np.zeros(
        len(nonlenses), dtype=np.int32)), axis=-1)

    lenses = glob.glob(os.path.join(
        os.getcwd(), path, dataset_name, 'Lenses', '*'))
    lenses = np.array(lenses)
    lenses = np.stack(
        (lenses, np.ones(len(lenses), dtype=np.int32)), axis=-1)

    filenames = np.concatenate((nonlenses, lenses))

    return filenames


def train_val_split(FILENAMES, seed=152):
    train, val = train_test_split(
        FILENAMES, train_size=0.9, test_size=0.1, random_state=seed)
    return train, val

# def fetch_train_val_filenames(path, dataset_name, nonlenses_dataset_size=None, lenses_dataset_size=None, seed=152):
#     rand = random.Random(seed)
#     nonlenses = glob.glob(os.path.join(
#         os.getcwd(), path, dataset_name, 'NonLenses', '*'))
#     nonlenses.sort()
#     rand.shuffle(nonlenses)
    
#     if nonlenses_dataset_size is not None:
#         nonlenses = nonlenses[:nonlenses_dataset_size]

#     train_nonlenses, val_nonlenses = train_val_split(nonlenses)
    
#     rand = random.Random()
#     lenses = glob.glob(os.path.join(
#         os.getcwd(), path, dataset_name, 'Lenses', '*'))
#     lenses.sort()
#     rand.shuffle(lenses)

#     if lenses_dataset_size is not None:
#         lenses = lenses[:lenses_dataset_size]
         
#     val_nonlenses = np.array(val_nonlenses)
#     val_nonlenses = np.stack((val_nonlenses, np.zeros(
#         len(val_nonlenses), dtype=np.int32)), axis=-1)            
    
#     val_lenses = lenses[:len(val_nonlenses)]
#     val_lenses = np.array(val_lenses)
#     val_lenses = np.stack(
#         (val_lenses, np.ones(len(val_lenses), dtype=np.int32)), axis=-1)
    
#     train_nonlenses = np.array(train_nonlenses)
#     train_nonlenses = np.stack((train_nonlenses, np.zeros(
#         len(train_nonlenses), dtype=np.int32)), axis=-1)

#     lenses = lenses[len(val_nonlenses):]
#     train_lenses = lenses[:len(train_nonlenses)]
#     train_lenses = np.array(train_lenses)
#     train_lenses = np.stack(
#         (train_lenses, np.ones(len(train_lenses), dtype=np.int32)), axis=-1)

#     train_filenames = np.concatenate((train_nonlenses, train_lenses))
#     np.random.shuffle(train_filenames)

#     val_filenames = np.concatenate((val_nonlenses, val_lenses))
#     np.random.shuffle(val_filenames)

#     return train_filenames, val_filenames


def fetch_train_val_filenames(path, dataset_name, nonlenses_dataset_size=None, lenses_dataset_size=None, seed=152):
    #FINAL VERSION.
    rand = random.Random(seed)
    nonlenses = glob.glob(os.path.join(
        os.getcwd(), path, dataset_name, 'NonLenses', '*'))
    nonlenses.sort()
    rand.shuffle(nonlenses)
    
    if nonlenses_dataset_size is not None:
        nonlenses = nonlenses[:nonlenses_dataset_size]

    train_nonlenses, val_nonlenses = train_val_split(nonlenses)
    
    lenses = glob.glob(os.path.join(
        os.getcwd(), path, dataset_name, 'Lenses', '*'))
    
    lenses_dict = defaultdict(lambda: None)
    for filepath in lenses:
        objid = filepath.split('/')[-1].split('_')[0]
        lenses_dict[objid] = filepath

    val_lenses = []
    
    for filepath in nonlenses[:len(val_nonlenses)]:
        objid = filepath.split('/')[-1].split('_')[0]
        
        if objid in lenses_dict:
            val_lenses.append(lenses_dict.pop(objid))
            
    val_nonlenses = np.array(val_nonlenses)
    val_nonlenses = np.stack((val_nonlenses, np.zeros(
        len(val_nonlenses), dtype=np.int32)), axis=-1)            
    
    val_lenses = np.array(val_lenses)
    val_lenses = np.stack(
        (val_lenses, np.ones(len(val_lenses), dtype=np.int32)), axis=-1)
    
    lenses = [lenses_dict[key] for key in lenses_dict]
    lenses.sort()
    rand = random.Random(seed)
    rand.shuffle(lenses)

    if lenses_dataset_size is not None:
        lenses = lenses[:lenses_dataset_size]
    
    rand = random.Random()
    rand.shuffle(lenses)    

    train_lenses = lenses[:len(train_nonlenses)]
    
    train_nonlenses = np.array(train_nonlenses)
    train_nonlenses = np.stack((train_nonlenses, np.zeros(
        len(train_nonlenses), dtype=np.int32)), axis=-1)

    train_lenses = np.array(train_lenses)
    train_lenses = np.stack(
        (train_lenses, np.ones(len(train_lenses), dtype=np.int32)), axis=-1)

    train_filenames = np.concatenate((train_nonlenses, train_lenses))
    np.random.shuffle(train_filenames)

    val_filenames = np.concatenate((val_nonlenses, val_lenses))
    np.random.shuffle(val_filenames)

    return train_filenames, val_filenames

# def fetch_train_val_filenames(path, dataset_name, nonlenses_dataset_size=None, lenses_dataset_size=None, seed=152):
#     #PERFORMANCE SCALING LENSES SET
#     rand = random.Random(seed)
#     nonlenses = glob.glob(os.path.join(
#         os.getcwd(), path, dataset_name, 'NonLenses', '*'))
#     nonlenses.sort()
#     rand.shuffle(nonlenses)
    
#     if nonlenses_dataset_size is not None:
#         nonlenses = nonlenses[:nonlenses_dataset_size]

#     train_nonlenses, val_nonlenses = train_val_split(nonlenses)
    
#     lenses = glob.glob(os.path.join(
#         os.getcwd(), path, dataset_name, 'Lenses', '*'))
    
#     lenses_dict = defaultdict(lambda: None)
#     for filepath in lenses:
#         objid = filepath.split('/')[-1].split('_')[0]
#         lenses_dict[objid] = filepath

#     val_lenses = []
    
#     for filepath in nonlenses[:len(val_nonlenses)]:
#         objid = filepath.split('/')[-1].split('_')[0]
        
#         if objid in lenses_dict:
#             val_lenses.append(lenses_dict.pop(objid))
            
#     val_nonlenses = np.array(val_nonlenses)
#     val_nonlenses = np.stack((val_nonlenses, np.zeros(
#         len(val_nonlenses), dtype=np.int32)), axis=-1)            
    
#     val_lenses = np.array(val_lenses)
#     val_lenses = np.stack(
#         (val_lenses, np.ones(len(val_lenses), dtype=np.int32)), axis=-1)
    
#     lenses = []

#     for filepath in nonlenses[len(val_nonlenses):]:
#         objid = filepath.split('/')[-1].split('_')[0]
        
#         if objid in lenses_dict:
#             lenses.append(lenses_dict.pop(objid))

#     remaining_lenses = [lenses_dict[key] for key in lenses_dict]
#     remaining_lenses.sort()
#     rand = random.Random()
#     rand.shuffle(remaining_lenses)

#     lenses.extend(remaining_lenses)
#     if lenses_dataset_size is not None:
#         lenses = lenses[:lenses_dataset_size]

#     train_lenses = lenses[:len(train_nonlenses)]
    
#     train_nonlenses = np.array(train_nonlenses)
#     train_nonlenses = np.stack((train_nonlenses, np.zeros(
#         len(train_nonlenses), dtype=np.int32)), axis=-1)

#     train_lenses = np.array(train_lenses)
#     train_lenses = np.stack(
#         (train_lenses, np.ones(len(train_lenses), dtype=np.int32)), axis=-1)

#     train_filenames = np.concatenate((train_nonlenses, train_lenses))
#     np.random.shuffle(train_filenames)

#     val_filenames = np.concatenate((val_nonlenses, val_lenses))
#     np.random.shuffle(val_filenames)

#     return train_filenames, val_filenames