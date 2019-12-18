import sys
sys.path.append('../')
import argparse
import datetime
import json
import os
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import globals
from globals import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, \
                    NUM_CHANNELS, NUM_CLASSES, EPOCHS, INPUT_SHAPE, \
                    K_NEGATIVE_SAMPLE_RATIO_WEIGHT, LEARNING_RATE, \
                    IMG_CAM_WIDTH, IMG_CAM_HEIGHT, NUM_CAM_CHANNELS, INPUT_SHAPE_CAM

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, ReduceLROnPlateau
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Input, concatenate, Reshape, BatchNormalization, Activation, Lambda, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from common.camera_model import CameraModel
from process.globals import CAM_IMG_BOTTOM, CAM_IMG_TOP
from loader import get_data_and_ground_truth, data_generator_train, \
                   data_number_of_batches_per_epoch, filter_camera_data_and_gt, \
                   generate_index_list, file_prefix_for_timestamp, \
                   load_data
from model import build_model, load_model
from pretrain import calculate_population_weights
from common import pr_curve_plotter
from common.csv_utils import foreach_dirset
from train import LossHistory


def load_fcn(model_file, weights_file, lockLidarModel, lockCameraModel, trainable):

    with open(model_file, 'r') as jfile:
        print('Loading weights file {}'.format(weights_file))
        print("reading existing model and weights")
        model = keras.models.model_from_json(json.loads(jfile.read()))
        model.load_weights(weights_file)
        
        for layer in model.layers:
            layer.trainable = trainable
            if (lockCameraModel and "cameraNet" in layer.name) or (lockLidarModel and "lidarNet" in layer.name):
                layer.trainable = False
                print layer.name, "trainable mode changed to False"
      
        model.compile(optimizer=Adam(lr=LEARNING_RATE),
                      loss="mean_squared_error", metrics=['mae'])
                                     
    print(model.summary())  
    return model

def load_gt(indicies, centroid_rotation, gt, obs_size):
    
    batch_index = 0

    for ind in indicies:
        gt[batch_index,0] = centroid_rotation[0][ind] #tx
        gt[batch_index,1] = centroid_rotation[1][ind] #ty
        gt[batch_index,2] = centroid_rotation[2][ind] #tz
        gt[batch_index,3] = centroid_rotation[5][ind] #rz
        gt[batch_index,4] = obs_size[0][ind]
        gt[batch_index,5] = obs_size[1][ind]
        gt[batch_index,6] = obs_size[2][ind]        
        batch_index += 1

def load_radar_data(indicies, radar_ranges_angles, radar_data):

    batch_index = 0
    for ind in indicies:
        radar_ranges_angles[batch_index,0] = radar_data[0][ind]
        radar_ranges_angles[batch_index,1] = radar_data[1][ind]
        batch_index +=1
        
def data_generator_FCN(obs_centroids, obs_size, 
                        pickle_dir_and_prefix_cam, pickle_dir_and_prefix_lidar,
                        batch_size, radar_data, cache):
                        
    tx = obs_centroids[0]
    ty = obs_centroids[1]
    tz = obs_centroids[2]
    rx = obs_centroids[3]
    ry = obs_centroids[4]
    rz = obs_centroids[5]
    
    obsl = obs_size[0]
    obsw = obs_size[1]
    obsh = obs_size[2]

    cam_images = np.ndarray(shape=(batch_size, globals.IMG_CAM_HEIGHT, 
                    globals.IMG_CAM_WIDTH, globals.NUM_CAM_CHANNELS), dtype=float)    
    lidar_images = np.ndarray(shape=(batch_size, globals.IMG_HEIGHT, 
                    globals.IMG_WIDTH, globals.NUM_CHANNELS), dtype=float)
    centroid_rotation_size = np.ndarray(shape=(batch_size, 7), dtype=float)
    centroid = np.ndarray(shape=(batch_size, 3), dtype=float)
    rz = np.ndarray(shape=(batch_size,1), dtype=float)
    radar_ranges_angles = np.ndarray(shape=(batch_size, 2), dtype=float)

    num_batches = data_number_of_batches_per_epoch(pickle_dir_and_prefix_cam, batch_size)

    indicies_list = np.arange(len(tx))
    
    is_cache_avail = False

#    if cache is not None:
#        is_cache_avail = cache['cam_images'] is not None and cache['centroid'] is not None
#        if not is_cache_avail:
#            cache['cam_images'] = np.ndarray(shape=(len(pickle_dir_and_prefix_cam), globals.IMG_CAM_HEIGHT, 
#                    globals.IMG_CAM_WIDTH, globals.NUM_CAM_CHANNELS), dtype=float)
#            cache['lidar_images'] = np.ndarray(shape=(len(pickle_dir_and_prefix_lidar), globals.IMG_HEIGHT, 
#                    globals.IMG_WIDTH, globals.NUM_CHANNELS), dtype=float)
#            cache['radar_data'] = np.ndarray(shape=(len(radar_data[0]), 2), dtype=float)        
#                    
#            cache['centroid'] = np.ndarray(shape=(len(obs_centroids[0]), 3), dtype=float)
#            cache['rz'] = np.ndarray(shape=(len(obs_centroids[0]),1), dtype=float)

    
    while 1:            

        if cache is not None:
            is_cache_avail = cache['cam_images'] is not None and cache['centroid'] is not None
            if not is_cache_avail:
                cache['cam_images'] = np.ndarray(shape=(len(pickle_dir_and_prefix_cam), globals.IMG_CAM_HEIGHT, 
                        globals.IMG_CAM_WIDTH, globals.NUM_CAM_CHANNELS), dtype=float)
                cache['lidar_images'] = np.ndarray(shape=(len(pickle_dir_and_prefix_lidar), globals.IMG_HEIGHT, 
                        globals.IMG_WIDTH, globals.NUM_CHANNELS), dtype=float)
                cache['radar_data'] = np.ndarray(shape=(len(radar_data[0]), 2), dtype=float)        
                        
                cache['centroid'] = np.ndarray(shape=(len(obs_centroids[0]), 3), dtype=float)
                cache['rz'] = np.ndarray(shape=(len(obs_centroids[0]),1), dtype=float)

        indicies = generate_index_list(indicies_list, True, num_batches, batch_size)

        for batch in range(num_batches):

            batch_indicies = indicies[batch * batch_size:batch * batch_size + batch_size]

            if not is_cache_avail:
                load_data(batch_indicies, lidar_images, pickle_dir_and_prefix_lidar, "lidar", globals.NUM_CHANNELS)
                load_data(batch_indicies, cam_images, pickle_dir_and_prefix_cam, "camera", globals.NUM_CAM_CHANNELS)
                load_radar_data(batch_indicies, radar_ranges_angles, radar_data) 
                load_gt(batch_indicies, obs_centroids, centroid_rotation_size, obs_size)
                np.copyto(centroid, centroid_rotation_size[:,0:3])
                np.copyto(rz, centroid_rotation_size[:,3:4]) 
                
                if cache is not None:
                    # save to cache
                    i = 0
                    for ind in batch_indicies:
                        np.copyto(cache['cam_images'][ind], cam_images[i])
                        np.copyto(cache['lidar_images'][ind], lidar_images[i])
                        np.copyto(cache['radar_data'][ind], radar_ranges_angles[i])
                        np.copyto(cache['centroid'][ind], centroid[i])
                        np.copyto(cache['rz'][ind], rz[i])
                        i += 1
                     
            else:
                # copy from cache
                i = 0
                for ind in batch_indicies:
                    np.copyto(cam_images[i], cache['cam_images'][ind])
                    np.copyto(lidar_images[i], cache['lidar_images'][ind])
                    np.copyto(radar_ranges_angles[i], cache['radar_data'][ind])
                    np.copyto(centroid[i], cache['centroid'][ind])
                    np.copyto(rz[i], cache['rz'][ind])
                    i += 1            
                        
                              
            yield ([cam_images, lidar_images, radar_ranges_angles], [centroid, rz])
   
    
def get_data_and_ground_truth_matching_lidar_cam_frames(csv_sources, parent_dir):
    
    txl_cam = []
    tyl_cam = []
    tzl_cam = []
    rxl_cam = []
    ryl_cam = []
    rzl_cam = []
    timestamps_cam = []
    obsl_cam = []
    obsw_cam = []
    obsh_cam = []
    
    pickle_dir_and_prefix_cam = []
    pickle_dir_and_prefix_lidar = []
    radar_range = []
    radar_angle = []

    def process(dirset):

        lidar_truth_fname = dirset.dir+"/obs_poses_interp_transform.csv"
        cam_truth_fname = dirset.dir+"/obs_poses_camera.csv"
        radar_data_fname = dirset.dir+"/radar/radar_tracks.csv"

        df_lidar_truths = pd.read_csv(lidar_truth_fname)
        lidar_truths_list = df_lidar_truths['timestamp'].tolist()
        
        df_radar_data = pd.read_csv(radar_data_fname)
         
        #print lidar_rows[:,'timestamp']   
        def nearest_lidar_timestamp(cam_ts):
            x = min(lidar_truths_list, key=lambda x:abs(x-cam_ts))
            return x
         
        def nearest_radar_timestamp_data(cam_ts):
            return df_radar_data.ix[(df_radar_data['timestamp']-cam_ts).abs().argsort()[0]]    

        with open(cam_truth_fname) as csvfile_2:
            readCSV_2 = csv.DictReader(csvfile_2, delimiter=',')

            for row2 in readCSV_2:
                ts = row2['timestamp']
                tx = row2['tx']
                ty = row2['ty']
                tz = row2['tz']
                rx = row2['rx']
                ry = row2['ry']
                rz = row2['rz']

                pickle_dir_prefix = file_prefix_for_timestamp(dirset.dir, "camera", ts)
                pickle_dir_and_prefix_cam.append(pickle_dir_prefix)
                txl_cam.append(float(tx))
                tyl_cam.append(float(ty))
                tzl_cam.append(float(tz))
                rxl_cam.append(float(rx))
                ryl_cam.append(float(ry))
                rzl_cam.append(float(rz))
                timestamps_cam.append(ts)
                obsl_cam.append(float(dirset.mdr['l']))
                obsw_cam.append(float(dirset.mdr['w']))
                obsh_cam.append(float(dirset.mdr['h']))
                lidar_ts = nearest_lidar_timestamp(int(ts))
                pickle_dir_prefix = file_prefix_for_timestamp(dirset.dir, "lidar", str(lidar_ts))
                pickle_dir_and_prefix_lidar.append(pickle_dir_prefix)
                
                radar_data = nearest_radar_timestamp_data(int(ts))
                radar_range.append(float(radar_data['range']))
                radar_angle.append(float(radar_data['angle']))
                              
                
    foreach_dirset(csv_sources, parent_dir, process)

    obs_centroid = [txl_cam, tyl_cam, tzl_cam, rxl_cam, ryl_cam, rzl_cam, timestamps_cam]
    obs_size = [obsl_cam, obsw_cam, obsh_cam]
    radar_data = [radar_range, radar_angle]
      
    
    return obs_centroid, pickle_dir_and_prefix_cam, obs_size, pickle_dir_and_prefix_lidar, radar_data


def build_FCN(input_layer, output_layer, net_name):

    # cam_net_out is too big. apply max_pooling
    if net_name=="cam2fcn":
        output_layer = MaxPooling2D(pool_size=(4, 1), strides=None, padding='valid', data_format=None)(output_layer)

    flatten_out = Flatten()(output_layer)
    dropout_1 = Dropout(0.2)(flatten_out)
    dense_1 = Dense(96, activation='relu', name='dense1_'+net_name,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dropout_1)
    dropout_2 = Dropout(0.2)(dense_1)             
    dense_2 = Dense(48, activation='relu', name='dense2_'+net_name,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dropout_2)
                   
    return dense_2

def build_FCN_cam_lidar(cam_inp, lidar_inp, cam_net_out, lidar_net_out,\
                        lockLidarModel, lockCameraModel):

    cam_net_out = build_FCN(cam_inp, cam_net_out, "cam2fcn")
    lidar_net_out = build_FCN(lidar_inp, lidar_net_out, "ldr2fcn")
    radar_inp = Input(shape=(2,), name='radar')
        
    dense = concat_normalized = concat_input = concatenate([cam_net_out, lidar_net_out, radar_inp])
    #concat_normalized = BatchNormalization(name='normalize', axis=-1)(concat_input)
    #dense = Dense(3, activation='relu', name='fcn.dense',
    #               kernel_initializer='random_uniform', bias_initializer='zeros')(concat_normalized)
    
    # output for centroid
    dense_1_1 = Dense(3, activation='elu', name='dense1_1', 
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense)                  
    dense_1_2 = Dense(3, activation='elu', name='dense1_2', 
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense)
    d_1 = Dense(3, activation='linear', name='d1')(concatenate([dense_1_1, dense_1_2]))
     
    # output for rotation   
    dense_2_1 = Dense(1, activation='elu', name='dense2_1', 
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense)                  
    dense_2_2 = Dense(1, activation='elu', name='dense2_2', 
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense)
    d_2 = Dense(1, activation='linear', name='d2')(concatenate([dense_2_1, dense_2_2]))


    model = Model(inputs=[cam_inp, lidar_inp, radar_inp], outputs=[d_1, d_2])     
    
    for layer in model.layers:
        layer.trainable = True
        if (lockCameraModel and "cameraNet" in layer.name) or (lockLidarModel and "lidarNet" in layer.name):
            layer.trainable = False
            print layer.name, "trainable mode changed to False"
                  
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss="mean_squared_error", metrics=['mae'])
 
   

    print(model.summary())
    return model
    
def main():

    parser = argparse.ArgumentParser(description='FCN trainer for radar/camera/lidar')
    parser.add_argument("--train_file", type=str, default="../data/train_folders.csv",
                        help="list of data folders for training")
    parser.add_argument("--val_file", type=str, default="../data/validation_folders.csv",
                        help="list of data folders for validation")
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--camera_model', type=str, default="", help='Model Filename')
    parser.add_argument('--camera_weights', type=str, default="", help='Weights Filename')
    parser.add_argument('--lidar_model', type=str, default="", help='Model Filename')
    parser.add_argument('--lidar_weights', type=str, default="", help='Weights Filename')
    parser.add_argument('--fcn_model', type=str, default="", help='Model Filename')
    parser.add_argument('--fcn_weights', type=str, default="", help='Weights Filename')
    parser.add_argument('--outdir', type=str, default="./", help='output directory')
    parser.add_argument('--cache', type=str, default=None, help='Cache data')
    
    # if fcn_model file is given, weights of all network will come from fcn model even if
    # camera/lidar model file names are given too. If you give camera/lidar model names along
    # with fcn model, camera/lidar part will not be trained, only the top part.
    
    # to create fcn model with previously trained camera/lidar models, do not enter fcn model 
    # file name. This way newly created fcn model will include camera/lidar weights and can be
    # used to train only top part next time you run training.

    args = parser.parse_args()
    train_file = args.train_file
    validation_file = args.val_file
    outdir = args.outdir
    dir_prefix = args.dir_prefix
           
       
    cache_train, cache_val = None, None
    if args.cache is not None:
        cache_train = {'cam_images': None, 'lidar_images': None, 'radar_data': None, 'centroid': None, 'rz': None}
        cache_val = {'cam_images': None, 'lidar_images': None, 'radar_data': None, 'centroid': None, 'rz': None}

    lockCameraModel = None   
    camera_net = None         
    if args.camera_model != "":  
        lockCameraModel = True
        if args.fcn_model == "":
            weightsFile = args.camera_model.replace('json', 'h5')
            if args.fcn_weights != "":
                weightsFile = args.camera_weights                 
            camera_net = load_model(args.camera_model, weightsFile,
                               INPUT_SHAPE_CAM, NUM_CLASSES, trainable=False,
                               layer_name_ext="cameraNet")
    else:  
        lockCameraModel = False     
        if args.fcn_model == "":           
            camera_net = build_model(
                               INPUT_SHAPE_CAM, NUM_CLASSES, trainable=True,
                               data_source="camera", layer_name_ext="cameraNet")    
    if camera_net is not None:                                          
        cam_inp_layer = camera_net.input
        cam_out_layer = camera_net.get_layer("deconv6acameraNet").output
    
    lockLidarModel = None
    lidar_net = None
    if args.lidar_model != "":
        lockLidarModel = True
        if args.fcn_model == "":
            weightsFile = args.lidar_model.replace('json', 'h5')
            if args.lidar_weights != "":
                weightsFile = args.lidar_weights
            lidar_net = load_model(args.lidar_model, weightsFile,
                               INPUT_SHAPE, NUM_CLASSES, trainable=False,
                               layer_name_ext="lidarNet")
    else:     
        lockLidarModel = False   
        if args.fcn_model == "":                
            lidar_net = build_model(
                               INPUT_SHAPE, NUM_CLASSES, trainable=True,
                               data_source="lidar", layer_name_ext="lidarNet")
    
    if lidar_net is not None:                   
        lidar_inp_layer = lidar_net.input
        lidar_out_layer = lidar_net.get_layer("deconv6alidarNet").output
    
    if args.fcn_model != "":
        weightsFile = args.fcn_model.replace('json', 'h5')
        if args.fcn_weights != "":
            weightsFile = args.fcn_weights
        cam_lidar_radar_net = load_fcn(args.fcn_model, weightsFile, \
                          lockLidarModel, lockCameraModel, True)
    else:    
        cam_lidar_radar_net = build_FCN_cam_lidar(cam_inp_layer, lidar_inp_layer, 
                        cam_out_layer, lidar_out_layer, lockLidarModel, lockCameraModel)
        # save the model
        with open(os.path.join(outdir, 'fcn_model.json'), 'w') as outfile:
            json.dump(cam_lidar_radar_net.to_json(), outfile)   


                                
    train_data = get_data_and_ground_truth_matching_lidar_cam_frames(train_file, dir_prefix)
    val_data = get_data_and_ground_truth_matching_lidar_cam_frames(validation_file, dir_prefix)

    # number of batches per epoch
    n_batches_per_epoch_train = data_number_of_batches_per_epoch(train_data[1], BATCH_SIZE)
    n_batches_per_epoch_val = data_number_of_batches_per_epoch(val_data[1], BATCH_SIZE)


    if args.fcn_model != "" and (args.camera_model != "" or args.lidar_model):
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"   
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print "!!!!                                                                      !!!!"
        print "!!!! YOU ENTERED CAMERA AND/OR LIDAR MODEL NAME ALONG WITH FCN MODEL NAME !!!!"
        print "!!!!         THIS WILL ONLY FREEZE THE TRAINING OF CAMERA/LIDAR           !!!!"
        print "!!!!        CAMERA/LIDAR WEIGHTS ARE STILL LOADED FROM FCN MODEL          !!!!"
        print "!!!!                                                                      !!!!"
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"



    print("Number of batches per epoch: {}".format(n_batches_per_epoch_train))
    print("start time:")
    print(datetime.datetime.now())

    checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'fcn_weights.{epoch:02d}-{loss:.4f}.hdf5'),
                                   verbose=1, save_weights_only=True)
    tensorboard = TensorBoard(histogram_freq=1, log_dir=os.path.join(outdir, 'tensorboard/'),
                              write_graph=True, write_images=False)
    loss_history = LossHistory()
    lr_schedule = ReduceLROnPlateau(monitor='d1_mean_absolute_error', factor=0.2, patience=3, verbose=1, \
                                mode='auto', epsilon=.2, cooldown=0, min_lr=0.0000001)

    try:
        if args.cache is None or args.cache == 'generate':
            cam_lidar_radar_net.fit_generator(
                data_generator_FCN(
                    train_data[0], train_data[2], train_data[1], 
                    train_data[3], BATCH_SIZE, train_data[4],
                    cache=cache_train
                ),  # generator
                n_batches_per_epoch_train,  # number of batches per epoch
                validation_data=data_generator_FCN(
                    val_data[0], val_data[2], val_data[1], 
                    val_data[3], BATCH_SIZE, val_data[4],
                    cache=cache_val
                ),
                validation_steps=n_batches_per_epoch_val,  # number of batches per epoch
                epochs=EPOCHS,
                callbacks=[checkpointer, tensorboard, loss_history, lr_schedule],
                verbose=1
            )
            
        elif args.cache == 'shuffle':
            # load all batches at once
            
            print(len(train_data[0][0]), len(val_data[0][0]))
            
            next(data_generator_FCN(
                train_data[0], train_data[2], train_data[1], 
                train_data[3], len(train_data[0][0]), train_data[4],
                cache=cache_train
            ))
            next(data_generator_FCN(
                val_data[0], val_data[2], val_data[1], 
                val_data[3], len(val_data[0][0]), val_data[4],
                cache=cache_val
            ))

            print len(cache_train['cam_images']), len(cache_train['centroid'])
            print len(cache_val['cam_images']), len(cache_val['centroid'])
            
            cam_lidar_radar_net.fit([cache_train['cam_images'], cache_train['lidar_images'], cache_train['radar_data']], 
                      [cache_train['centroid'], cache_train['rz']],
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=1,
                      callbacks=[checkpointer, loss_history, lr_schedule],
                      validation_data=([cache_val['cam_images'], cache_val['lidar_images'], cache_val['radar_data']], \
                      [cache_val['centroid'], cache_val['rz']]),
                      shuffle=True)
   
            
        
    except KeyboardInterrupt:
        print('\n\nExiting training...')

    print("stop time:")
    print(datetime.datetime.now())
    # save model weights
    cam_lidar_radar_net.save_weights(os.path.join(outdir, "fcn_model.h5"), True)

    #print precision_recall_array -- NO MORE PRECISON/RECALL -- GIVES ERROR IF UNCOMMENTED
    #pr_curve_plotter.plot_pr_curve(loss_history, outdir)


if __name__ == '__main__':
    main()
