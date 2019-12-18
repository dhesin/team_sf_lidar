#!/usr/bin/python
"""transform lidar cloud points with rotations and/or translations
"""
import math
import numpy as np
import random
from tf.transformations import rotation_matrix, translation_matrix, concatenate_matrices

'''
    Transform a point cloud by rotation around z-axis and translation in (x, y) space.    
    rotation: rotation angle in radians. Default: None, random rotation
    translation: (trans_x, trans_y), a tuple of translation amount in x and y. Default: None, random translation
    Returns the new point cloud.
'''
def transform(points, rotation=None, translation=None):    
    if rotation == None:
        rotation = (random.random()*360.0 - 180) /180 * math.pi   
    print('rotation={}'.format(rotation))
    direction = np.array([0, 0, 1])
    R = rotation_matrix(rotation, direction)  
    
    if translation == None:        
        translation_dir = np.random.random(2)*6 - 3 
        translation_dir = np.append(translation_dir, 0)
    else:
        translation_dir = np.array([translation[0], translation[1], 0])  
    print('translation={}'.format(translation_dir))       
    T = translation_matrix(translation_dir)
    
    M = concatenate_matrices(R, T)     
    points_4d = points[:,:4].copy()    
    points_4d[:,3] = np.ones(points.shape[0])        
    points_4d =  np.dot(points_4d, M.T)
    points[:,:3] = points_4d[:,:3]        
    
    return points
    
'''
    Random transformations of a point cloud for a number of times.
    Returns a list of point clouds.
'''    
def transform_n(points, n=1):
    res = []
    for i in range(n):
        res.append(transform(points))
       
    return res
