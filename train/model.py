
# coding: utf-8

import os
import numpy as np
from numpy import random
import cv2
import keras
import json
from keras import backend as K
import tensorflow as tf
import globals

from keras.layers import Input, concatenate, Reshape, BatchNormalization, Activation, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

# Disabling both USE_W1 and USE_W2 should result in typical categorical_cross_entropy loss
USE_W1 = True
USE_W2 = True


    
def custom_weighted_loss(input_shape, use_regression, weight_bb, obj_to_bkg_ratio=0.00016, avg_obj_size=1000, loss_scaler=1000):

    def custom_loss(y_true, y_pred):
        # the code here is only executed once, since these should all be graph operations. Do not expect Numpy
        # calculations or the like to work here, only Keras backend and tensor flow nodes.

        max_pixels = input_shape[0] * input_shape[1]
        #y_pred = tf.Print(y_pred, ["y_pred", tf.shape(y_pred)])
        
        if use_regression: 
            y_true_obj, y_true_bb = tf.split(y_true, [globals.NUM_CLASSES, globals.NUM_REGRESSION_OUTPUTS], 2)
            y_pred_obj, y_pred_bb = tf.split(y_pred, [globals.NUM_CLASSES, globals.NUM_REGRESSION_OUTPUTS], 2)            
        else:
            y_true_obj = y_true
            y_pred_obj = y_pred 
        

        log_softmax = tf.log(y_pred_obj, name="logsoftmax")        
        neglog_softmax = tf.scalar_mul(-1., log_softmax)

        pixel_loss = tf.multiply(y_true_obj, neglog_softmax, name="pixel_loss")       
        
        labels_bkg, labels_frg = tf.split(y_true_obj, 2, 2, name="split_2")
        bkg_frg_areas = tf.reduce_sum(y_true_obj, 1)
        bkg_area, frg_area = tf.split(bkg_frg_areas, 2, 1, name="split_1")

        # The branches here configure the graph differently. You can imagine these branches working as if the path
        # that was disabled didn't exist at all in the code. Each path should work independently.
        
        if USE_W1:
            w1_bkg_weights = tf.scalar_mul(obj_to_bkg_ratio, labels_bkg)
        else:
            w1_bkg_weights = labels_bkg
               
        frg_area_tiled = tf.tile(frg_area, tf.stack([1, max_pixels]))

        # prevent divide by zero, max is number of pixels
        frg_area_tiled = K.clip(frg_area_tiled, K.epsilon(), max_pixels)
        inv_frg_area = tf.div(tf.ones_like(frg_area_tiled), frg_area_tiled)

        w2_weights = tf.scalar_mul(avg_obj_size, inv_frg_area)
        w2_frg_weights = tf.multiply(labels_frg, tf.expand_dims(w2_weights, axis=2))                                  

        w1_times_w2 = tf.add(w1_bkg_weights, w2_frg_weights, name="w1_times_w2")
        weighted_loss = tf.multiply(w1_times_w2, pixel_loss, name="weighted_loss")        
        weighted_loss = tf.scalar_mul(loss_scaler, weighted_loss)

        loss = loss_obj = tf.reduce_sum(weighted_loss, -1, name="loss")                          
        
        # weighted loss for regression branch
        if use_regression: 
            diff_bb = tf.subtract(y_true_bb, y_pred_bb)       
            l2_norm = tf.norm(diff_bb)       
            
            weighted_loss_bb = tf.multiply(w2_frg_weights, l2_norm, name="weighted_l2_loss")
            loss_bb = tf.reduce_sum(weighted_loss_bb, -1, name="loss_bb")        
            loss_bb = tf.scalar_mul(weight_bb, loss_bb)
            #loss_obj = tf.Print(loss_obj, ["loss_obj max", tf.reduce_max(loss_obj), " mean:", tf.reduce_mean(loss_obj)])
            #loss_bb = tf.Print(loss_bb, ["loss_bb max:", tf.reduce_max(loss_bb), " mean:", tf.reduce_mean(loss_bb)])  
            
            loss = tf.add(loss_obj, loss_bb, name="loss")
            
        #loss = tf.Print(loss, ["loss", tf.shape(loss), loss])        
        return loss
    
    return custom_loss

def build_model(input_shape, num_classes, data_source,
                use_regression=False,
                obj_to_bkg_ratio=0.00016,
                avg_obj_size=1000,
                weight_bb=0.01,
                metrics=None,
                trainable=True):

    # set channels last format
    K.set_image_data_format('channels_last')

    # vertical strides
    vs = 2
    if data_source == "lidar":
        vs = globals.LIDAR_CONV_VERTICAL_STRIDE

    post_normalized = inputs = Input(shape=input_shape, name='input')
    if globals.USE_SAMPLE_WISE_BATCH_NORMALIZATION:
        flatten_input = Reshape((-1, input_shape[2]), name='flatten_input')(inputs)
        normalized = BatchNormalization(name='normalize', axis=1)(flatten_input)
        post_normalized = Reshape((input_shape[0], input_shape[1], input_shape[2]), name='unflatten_input')(normalized)
    if globals.USE_FEATURE_WISE_BATCH_NORMALIZATION:
        post_normalized = BatchNormalization(name='normalize', axis=-1)(post_normalized)
    inputs_padded = ZeroPadding2D(padding=((0, 0), (0, 3)))(post_normalized)
    conv1 = Conv2D(4, 5, strides=(vs, 4), activation='relu', name='conv1', padding='same',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(inputs_padded)
    conv2 = Conv2D(6, 5, strides=(vs, 2), activation='relu', name='conv2', padding='same',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(conv1)
    conv3 = Conv2D(12, 5, strides=(vs, 2), activation='relu', name='conv3', padding='same',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(conv2)
    deconv4 = Conv2DTranspose(16, 5, strides=(vs, 2), activation='relu', name='deconv4', padding='same',
                              kernel_initializer='random_uniform', bias_initializer='zeros')(conv3)

    concat_deconv4 = concatenate([conv2, deconv4], name='concat_deconv4')

    # classification task
    deconv5a = Conv2DTranspose(8, 5, strides=(vs, 2), activation='relu', name='deconv5a', padding='same',
                               kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv4)

    deconv5a_padded = Cropping2D(cropping=((0, 0), (1, 0)))(deconv5a)

    concat_deconv5a = concatenate([conv1, deconv5a_padded], name='concat_deconv5a')
    deconv6a = Conv2DTranspose(2, 5, strides=(vs, 4), name='deconv6a', padding='same',
                               kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv5a)

    if data_source == "lidar":
        deconv6a_crop = Cropping2D(cropping=((0, 0), (0, 3)))(deconv6a)
    elif data_source == "camera":
        deconv6a_crop = Cropping2D(cropping=((0, 0), (0, 4)))(deconv6a)
    else:
        print "invalid data source"
        exit(1)
        
    deconv6a_flatten = Reshape((-1, num_classes), name='deconv6a_flatten')(deconv6a_crop)
    softmax = Activation('softmax',name='softmax')(deconv6a_flatten)
    output = classification_output = Lambda(lambda x: K.clip(x, K.epsilon(), 1), name='classification_output')(softmax)

    # regression task
    if use_regression:        
        deconv5b = Conv2DTranspose(globals.NUM_REGRESSION_OUTPUTS, 5, strides=(vs,2), activation='relu',
                                   name='deconv5b', padding='same',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv4)

        if data_source == "camera":
            deconv5b_padded = ZeroPadding2D(padding=((1, 0), (0, 0)))(deconv5b)
        elif data_source == "lidar":
            deconv5b_padded = Cropping2D(cropping=((0, 0), (1, 0)))(deconv5b)
        else:
            print "invalid data source"
            exit(1)
                
            
        concat_deconv5b = concatenate([conv1, deconv5b_padded], name='concat_deconv5b')
        
        
        deconv6b = Conv2DTranspose(globals.NUM_REGRESSION_OUTPUTS, 5, strides=(vs,4), activation='relu',
                                   name='deconv6b', padding='same',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv5b)
        
        if data_source == "camera":
            deconv6b_crop = Cropping2D(cropping=((3, 0), (0, 4)))(deconv6b)
        elif data_source == "lidar":
            deconv6b_crop = Cropping2D(cropping=((0, 0), (0, 3)))(deconv6b)
        else:
            print "invalid data source"
            exit(1)
            
        regression_output = Reshape((-1, globals.NUM_REGRESSION_OUTPUTS), name='regression_output')(deconv6b_crop)
        
        # concatenate two outputs into one so that we can have one loss function       
        output = concatenate([classification_output, regression_output], name='outputs')        
 
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=globals.LEARNING_RATE),
                  loss=custom_weighted_loss(input_shape, use_regression, weight_bb, obj_to_bkg_ratio, avg_obj_size),
                  metrics=metrics)
       
    print(model.summary())

    return model


def load_model(model_file, weights_file, input_shape, num_classes, use_regression,
               obj_to_bkg_ratio=0.00016,
               avg_obj_size=1000,
               weight_bb=0.01,
               metrics=None):
    with open(model_file, 'r') as jfile:
        print('Loading weights file {}'.format(weights_file))
        print("reading existing model and weights")
        model = keras.models.model_from_json(json.loads(jfile.read()))
        model.load_weights(weights_file)
        model.compile(optimizer=Adam(lr=globals.LEARNING_RATE),
                      loss=custom_weighted_loss(input_shape, use_regression, weight_bb, obj_to_bkg_ratio, avg_obj_size),
                      metrics=metrics)

    return model


def test(model):
    # please change path if needed
    path = '../../dataset1/10/lidar_360/1490991699437114271_'
    if os.path.exists(path + 'height.png'):
        print('image found')
    height_img = cv2.imread(path + 'height.png') 
    height_gray = cv2.cvtColor(height_img, cv2.COLOR_RGB2GRAY)
    distance_img = cv2.imread(path + 'distance.png') 
    distance_gray = cv2.cvtColor(distance_img, cv2.COLOR_RGB2GRAY)
    intensity_img = cv2.imread(path + 'intensity.png') 
    intensity_gray = cv2.cvtColor(intensity_img, cv2.COLOR_RGB2GRAY)
    x = np.zeros(globals.INPUT_SHAPE)
    x[:, :, 0] = height_gray
    x[:, :, 1] = distance_gray
    x[:, :, 2] = intensity_gray

    label = np.zeros(globals.INPUT_SHAPE[:2])
    label[8:, 1242:1581] = 1  #bounding box of the obstacle vehicle
    y = to_categorical(label, num_classes=2) #1st dimension: off-vehicle, 2nd dimension: on-vehicle
    #print(np.nonzero(y[:,1])[0].shape[0])
    
    #place holder
    regression_label = np.zeros((1, globals.IMG_WIDTH*globals.IMG_HEIGHT, 24))    
    outputs = np.concatenate((np.asarray([y]), regression_label), axis=2)
    print(outputs.shape)
    model.fit(np.asarray([x]), outputs, batch_size=globals.BATCH_SIZE, epochs=globals.EPOCHS, verbose=1)


def main():
    model = build_model(globals.INPUT_SHAPE, 2, use_regression=True)
    test(model)
    
if __name__ == '__main__':
    main()



