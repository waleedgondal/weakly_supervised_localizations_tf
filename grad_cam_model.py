# -*- coding: utf-8 -*-
"""
Utility functions.

Created on Fri Feb 10 19:30:28 2017
@author: M. Waleed Gondal
"""

import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

class Grad_CAM():
    
    """Grad_CAM (Gradient based Class Activation Maps) implements a weakly supervised localization method proposed at NIP 2016
    workshop. https://arxiv.org/abs/1610.02391
    
    The approach is published under the name of 'Grad-CAM: Why did you say that? Visual Explanations from Deep Networks 
    via Gradient-based Localization', Bolei Zhou et al.
    
    The class defines VGG16 model. The class initializes the network with pretrained weights for VGG16. 
    
    Parameters
    -----------
    n_labels: int
        An integer representing the number of output classes    
    weight_file_path: list of numpy arrays
        List of arrays that contain pretrained network parameters. The layers are parsed with respect
        to the respective layer name.
    
    Yields
    --------
    output : numpy array of float32
        The corresponding output scores. Shape (batchsize, num_labels)""" 
    
    def __init__(self, n_labels, weight_file_path = None):
        self.image_mean = [103.939, 116.779, 123.68]
        self.n_labels = n_labels       
        self.epsilon = 1e-4
        self.g = tf.get_default_graph()    
        assert (weight_file_path is not None), 'No weight file found'
        self.pretrained_weights = np.load(weight_file_path, encoding='bytes')

    def get_conv_weight( self, name ):
        """Accessing conv biases from the network weights"""
        return (self.pretrained_weights[name])            
        
    def get_conv_bias( self, name ):
        """Accessing biases from the network weights"""
        return (self.pretrained_weights[name])

    def conv_layer( self, bottom, name, stride = 1):
        
        """Implementation of convolutional layer using tensorflow predefined conv function.  
        Parameters
        ----------
        bottom: Tensor
            A tensor of shape (batchsize, height, width, channels)
        name: String
            A name for the variable scope according to the layer it belongs to.
        stride: Int
            An integer value defining the convolutional layer stride.
            
        Yields
        --------
        relu : Tensor
            A tensor of shape (batchsize, height, width, channels)"""        
              
        with tf.variable_scope(name) as scope:
            # The weights are retrieved according to how they are stored in arrays
            w = self.get_conv_weight(name+'_W')
            b = self.get_conv_bias(name+'_b')
            conv_weights = tf.get_variable(
                    "W",
                    shape=w.shape,
                    initializer=tf.constant_initializer(w)
                    )
            conv_biases = tf.get_variable(
                    "b",
                    shape=b.shape,
                    initializer=tf.constant_initializer(b)
                    )

            conv = tf.nn.conv2d( bottom, conv_weights, [1,stride,stride,1], padding='SAME')
            bias = tf.nn.bias_add( conv, conv_biases)
            
            with self.g.gradient_override_map({'Relu': 'GuidedRelu'}):
                relu = tf.nn.relu(bias, name=name)
        return relu  

    def network( self, image, is_training = True, dropout = 1.0): 
        
        """ Defines the Standard VGG16 Network, proposed in Very Deep Convolutional Networks for 
        Large-Scale Image Recognition Simonyan et al. https://arxiv.org/abs/1409.1556
        
        Parameters
        ----------
        image: Tensor
            A tensor of shape (batchsize, height, width, channels)
            
        Yields
        --------
        output : numpy array of float32
            The corresponding output scores. Shape (batchsize, num_labels)"""                    

        image *= 255.
        r, g, b = tf.split(3, 3, image)
        image = tf.concat(3,
            [
                b-self.image_mean[0],
                g-self.image_mean[1],
                r-self.image_mean[2]
            ])
        
        relu1_1 = self.conv_layer( image, "conv1_1" )
        relu1_2 = self.conv_layer( relu1_1, "conv1_2" )
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        relu2_1 = self.conv_layer(pool1, "conv2_1")
        relu2_2 = self.conv_layer(relu2_1, "conv2_2")
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        relu3_1 = self.conv_layer( pool2, "conv3_1")
        relu3_2 = self.conv_layer( relu3_1, "conv3_2")
        relu3_3 = self.conv_layer( relu3_2, "conv3_3")
        pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        relu4_1 = self.conv_layer( pool3, "conv4_1")
        relu4_2 = self.conv_layer( relu4_1, "conv4_2")
        relu4_3 = self.conv_layer( relu4_2, "conv4_3")
        pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        relu5_1 = self.conv_layer( pool4, "conv5_1")
        relu5_2 = self.conv_layer( relu5_1, "conv5_2")
        relu5_3 = self.conv_layer( relu5_2, "conv5_3")        
        pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5') 
            
        with tf.variable_scope('fc1') as scope:                        
            w = self.get_conv_weight('fc6_W')
            b = self.get_conv_bias('fc6_b')
            fc_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            fc_biases  = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))           
            shape = int(np.prod(pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc_weights), fc_biases)
            with self.g.gradient_override_map({'Relu': 'GuidedRelu'}):
                self.fc1 = tf.nn.relu(fc1l)
            self.fc1 = tf.nn.dropout(self.fc1, dropout)

        with tf.variable_scope('fc2') as scope:           
            w = self.get_conv_weight('fc7_W')
            b = self.get_conv_bias('fc7_b')
            fc_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            fc_biases  = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))           
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc_weights), fc_biases)
            with self.g.gradient_override_map({'Relu': 'GuidedRelu'}):
                self.fc2 = tf.nn.relu(fc2l)
            self.fc2 = tf.nn.dropout(self.fc2, dropout)

        with tf.variable_scope('fc3') as scope:
            w = self.get_conv_weight('fc8_W')
            b = self.get_conv_bias('fc8_b')
            fc_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            fc_biases  = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))         
            out = tf.nn.bias_add(tf.matmul(self.fc2, fc_weights), fc_biases)
            
        return out

    def get_grad_cam(self, fmaps, gradients, height = 224, width = 224, num_fmaps = 512):
    
        """ Computes Class Activation Maps using gradients backpropagated from output to the last convolutional
        layer in the network
        Parameters
        -----------
        fmaps: Numpy array of float32
            A batch of feature maps. Shape (batchsize, height, width, channels).
        gradients: Numpy array of float32
            Gradients computed via backpropagation. Shape (height, width, channels).
        height: Int
            An integer to which the CAM height is to be upsampled. It should be the height of input image.
        width: Int
            An integer to which the CAM width is to be upsampled. It should be the width of input image
        num_fmaps: Int
            Corresponds to the number of feature maps in the last convolutional layer. In simple terms it's the depth of
 
        Returns
        ---------
        Class Activation Map (CAM), a single channeled, upsampled, weighted sum of last conv filter maps. """
        
        weights = np.mean(gradients, axis=(0,1))

        fmaps_resized = tf.image.resize_bilinear(fmaps, [height, width] )
        fmaps_reshaped = tf.reshape(fmaps_resized, [-1, height*width, num_fmaps]) 
        label_w = tf.reshape( weights, [-1, num_fmaps, 1])
        classmap = tf.batch_matmul(fmaps_reshaped, label_w )
        classmap = tf.reshape( classmap, [-1, height, width] )
        return classmap    




