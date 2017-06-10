# -*- coding: utf-8 -*-
"""
Utility functions.

Created on Fri Oct 11 10:24:31 2016
@author: M. Waleed Gondal
"""
import numpy as np
import tensorflow as tf
import cv2


def normalize(img, s=0.1):
    """Normalize the image range for visualization"""
    z = img / np.std(img)
    return np.uint8(np.clip((z - z.mean()) / max(z.std(), 1e-4) * s + 0.5, 0, 1) * 255)

def threshold_cam(heatmap, threshold):
    """Exracts the highly relevant region (high intensity) in the heatmap 
    
    Parameters
    ----------
    heatmap : A 2D Float or Int array
        A 2D array of shape (height, width).
    threshold: Int 
        An integer value between 0 and 1, thresholding the heatmap to a certain percentage of the maximum value

    Yields
    --------
    image : numpy array of float32
        A binary image of shape (height, width). Where high intensity regions are represented by pixel value 1""" 
    
    # Binarize the heatmap
    _, thresholded_heatmap = cv2.threshold(heatmap, threshold * heatmap.max(), 1, cv2.THRESH_BINARY)
    # Required for converting image to uint8
    thresholded_heatmap = cv2.convertScaleAbs(thresholded_heatmap)
    return thresholded_heatmap

def draw_bbox(image, heatmap, threshold):
    
    """Extracts the bounding box from heatmap by thresholding it to a certain percentage of its maximum value
    
    Parameters
    ----------
    image : numpy array of float32
        Float array of shape (height, width, channels).
    heatmap : A 2D Float or Int array
        A 2D array of shape (height, width).
    threshold: Int 
        An integer value between 0 and 1, thresholding the heatmap to a certain percentage of the maximum value

    Yields
    --------
    image : numpy array of float32
        An image of shape (height, width, channels)."""
    
    img_copy = image.copy()
    thresholded_heatmap = threshold_cam(heatmap, threshold)
    contours, hier = cv2.findContours(thresholded_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = []   
    for i, c in enumerate(contours):
        contour_areas.append(cv2.contourArea(c))
    # Sort out the contours with respect to areas
    sorted_contours = sorted(zip(contour_areas, contours), key=lambda x:x[0], reverse=True)
    biggest_contour= sorted_contours[0][1]
    x,y,w,h = cv2.boundingRect(biggest_contour)
    cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0, 255,0), 2)
    return img_copy

def threshold_hmap(img, threshold=0.65):
    """ For removing slightly activated regions in prediction maps, it makes posterior map good"""
    
    img_copy = np.zeros(img.shape)
    print np.shape(img_copy)
    rows, cols = img.shape
    min_pixel = img.min()
    for i in range(rows):
        for j in range(cols):
            if(img[i,j] > 0.65):  # Intersection Region
                img_copy[i,j] = img[i,j]
            else:
                img_copy[i,j] = min_pixel
    return img_copy

def augment(image):
    """ Runtime Augmentations while training the network
    
    Parameters
    ----------
    image : Tensor
        A tensor of shape (height, width, channels).

    Yields
    --------
    image : A tensor
        A tensor of shape (height, width, channels)."""

    # Randomly flip the image
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    # Because these operations are not commutative, consider randomizing the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.05)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.6, upper=0.8)
    distorted_image = tf.image.central_crop(distorted_image, 0.95)
    
    return distorted_image
    
def save_to_csv(fnames, labels, csvpath):
    """Save a data set given by ``fnames`` and ``labels`` to a csv file.
    
    Parameters
    ----------
    fnames : list of strings
        List of file names.
    labels : list of ints or list of floats
        List of corresponding labels."""
    
    with open(csvpath, "w") as f:
        writer = csv.writer(f)
        for f, l in zip(fnames, labels):
            if isinstance(l, tuple):
                writer.writerow([f]+list(l))
            else:
                writer.writerow([f, l])
                
def draw_bbox_binary(image, binary_mask):
    
    """Extracts the bounding box from the binary mask. Returns an image with the biggest bounding box on it.
    
    Parameters
    ----------
    image : numpy array of float32
        Float array of shape (height, width, channels).
    binary_mask : A 2D Float or Int array
        A 2D array of shape (height, width).

    Yields
    --------
    image : numpy array of float32
        An image of shape (height, width, channels)."""
    
    img_copy = image.copy()
    contours, hier = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = []   
    for i, c in enumerate(contours):
        contour_areas.append(cv2.contourArea(c))
    # Sort out the contours with respect to areas
    sorted_contours = sorted(zip(contour_areas, contours), key=lambda x:x[0], reverse=True)
    biggest_contour= sorted_contours[0][1]
    x,y,w,h = cv2.boundingRect(biggest_contour)
    cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0, 255,0), 2)
    return img_copy 
                
def compute_froc(TP_list, FP_list, num_of_positives, count):
    
    """Generates the data required for plotting the froc curve
    
    Parameters
    ----------
    TP_list : List
        A list of probabilities corresponding to the truly detected regions in an image. 
    FP_list: List
        A list of probabilities corresponding to the falsely detected regions in an image.
    num_of_positives: Int
        Total number of objects present in the image.
        
    Yields
    --------
    total_sensitivity:  A list containig overall sensitivity of the system
    for different thresholds"""
    
    unlisted_FPs = [item for sublist in FP_list for item in sublist]
    unlisted_TPs = [item for sublist in TP_list for item in sublist] 
    
    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())    
    total_FPs.append(0)
    total_TPs.append(0)
    total_sensitivity = np.asarray(total_TPs)/float(num_of_positives)      
    return total_sensitivity
