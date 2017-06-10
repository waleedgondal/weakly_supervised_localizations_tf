# -*- coding: utf-8 -*-
"""
Utility functions.

Created on Fri Oct 11 10:24:31 2016
@author: M. Waleed Gondal
"""
import numpy as np
import tensorflow as tf
import cv2
import csv
from utils import augment


def make_batches(path_to_csv, n_epochs ,height , width, batch_size, training = True):
    
    """Make shuffled batches of images and their corresponding labels.
    
    Parameters
    ----------
    path_to_csv : string
        The path of csv file to be read.    
    n_epochs : int 
        The number of iterations for which the complete dataset is to be iterated.   
    height: int
        Height of an image
    width: int
        Width of an image
    batch_size: int
        The number of images to be stacked in one batch.
    training : Bool [default: True]
        
    Returns
    --------
    tf_label: Tensor
        A tensor of labels with shape (batchsize, label)
    tf_image: Tensor
        A tensor of image shape(batchsize, height, width, channels)"""
    
    image, label = read_data_from_csv(path_to_csv, n_epochs, height, width, training)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size, num_threads =10,
                                                      capacity = 1000 + 3*batch_size, min_after_dequeue = 1000)
    return image_batch, label_batch


def read_data_from_csv(path_to_csv, n_epochs, height, width, training = True):
    
    """Read filenames and labels from a csv file. The csv file has to contain
    one file name with complete path and one integer label, separated by comma. The
    implementation follows Tensorflow input pipeline.
    
    Parameters
    ----------
    path_to_csv : string
        The path of csv file to be read.
    
    n_epochs : int 
        The number of iterations for which the complete dataset is to be iterated.
    
    height: int
        Height of an image
    width: int
        Width of an image
    training : Bool [default: True]
        
    Returns
    --------
    tf_label: Tensor
        A tensor of labels with shape (batchsize, label)
    tf_image: Tensor
        A tensor of image shape(batchsize, height, width, channels)"""
    
    csv_path = tf.train.string_input_producer([path_to_csv], num_epochs=n_epochs)
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(csv_path)
    im_name, im_label = tf.decode_csv(csv_content, record_defaults=[[""], [1]])
    im_content = tf.read_file(im_name)
    tf_image = tf.image.decode_jpeg(im_content, channels=3)
    tf_image = tf.cast(tf_image, tf.float32) / 255.
    if training == True:
        tf_image = augment(tf_image)
    size = tf.cast([height, width], tf.int32)
    tf_image = tf.image.resize_images(tf_image, size)
    tf_label = tf.cast(im_label, tf.int32)
    
    return tf_image, tf_label

def count_images(file_path):
    """ Counts the number of rows (and hence the items) in a csv file"""
    file = open(file_path)
    reader = csv.reader(file)
    count = sum(1 for row in reader)
    file.close()
    return count

def batch_norm_wrapper(self, inputs, is_training, decay = 0.999):
    """Implementation of batch normalization for training.
    The wrapper is taken from http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    which is a simpler version of Tensorflow's 'official' version. See:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102"""

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2]) # [0,1,2] for global normalization as mentioned in documentation
            
        train_mean = tf.assign(pop_mean,
                              pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, self.epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, self.epsilon)