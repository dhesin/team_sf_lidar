import sys
import os
import argparse
import json
import datetime
import numpy as np
import cv2
import math
import csv
import rosbag
import sensor_msgs.point_cloud2
import keras
import pandas as pd
print(sys.path)
sys.path.append('../')
from common.camera_model import CameraModel
from process.globals import X_MIN, Y_MIN, RES, RES_RAD, LIDAR_MIN_HEIGHT
from scipy.ndimage.measurements import label
from globals import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, \
                    INPUT_SHAPE, BATCH_SIZE, PREDICTION_FILE_NAME, \
                    IMG_CAM_WIDTH, IMG_CAM_HEIGHT, NUM_CAM_CHANNELS, \
                    INPUT_SHAPE_CAM, LEARNING_RATE
from loader import get_data, data_number_of_batches_per_epoch, \
                   data_generator_train, data_generator_predict, \
                   file_prefix_for_timestamp
import model as model_module
from process.extract_rosbag_lidar import generate_lidar_2d_front_view

from keras.models import model_from_json
from keras.optimizers import Adam
from common.csv_utils import foreach_dirset
from train_fcn import data_generator_FCN, load_fcn
from predict import write_prediction_data_to_csv


def get_predict_data_matching_lidar_cam_frames(csv_sources, parent_dir):

    txl = []
    tyl = []
    tzl = []
    rxl = []
    ryl = []
    rzl = []
    
    obsl = []
    obsw = []
    obsh = []
        
    pickle_dir_and_prefix_cam = []
    pickle_dir_and_prefix_lidar = []
    radar_range = []
    radar_angle = []

    def process(dirset):
    
        timestamp_lidars = dirset.dir+"/lidar_timestamps.csv"
        timestamp_camera = dirset.dir+"/camera_timestamps.csv"
        radar_data_fname = dirset.dir+"/radar/radar_tracks.csv"

    
        df_lidar_timestamps = pd.read_csv(timestamp_lidars)
        lidar_timestamp_list = df_lidar_timestamps['timestamp'].tolist()
       
        df_radar_data = pd.read_csv(radar_data_fname)
         
        #print lidar_rows[:,'timestamp']   
        def nearest_lidar_timestamp(cam_ts):
            x = min(lidar_timestamp_list, key=lambda x:abs(x-cam_ts))
            return x
            
        def nearest_radar_timestamp_data(cam_ts):
            return df_radar_data.ix[(df_radar_data['timestamp']-cam_ts).abs().argsort()[0]]    

        with open(timestamp_camera) as csvfile_2:
            readCSV_2 = csv.DictReader(csvfile_2, delimiter=',')

            for row2 in readCSV_2:
                ts = row2['timestamp']
                txl.append(1.0)
                tyl.append(1.0)
                tzl.append(1.0)
                rxl.append(1.0)
                ryl.append(1.0)
                rzl.append(1.0)
                obsl.append(1.0)
                obsw.append(1.0)
                obsh.append(1.0)
 
                pickle_dir_prefix = file_prefix_for_timestamp(dirset.dir, "camera", ts)
                pickle_dir_and_prefix_cam.append(pickle_dir_prefix)

                lidar_ts = nearest_lidar_timestamp(int(ts))
                pickle_dir_prefix = file_prefix_for_timestamp(dirset.dir, "lidar", str(lidar_ts))
                pickle_dir_and_prefix_lidar.append(pickle_dir_prefix)
 
                radar_data = nearest_radar_timestamp_data(int(ts))               
                radar_range.append(float(radar_data['range']))
                radar_angle.append(float(radar_data['angle']))

                
    foreach_dirset(csv_sources, parent_dir, process)
      
    obs_centroid = [txl, tyl, tzl, rxl, ryl, rzl]
    obs_size = [obsl, obsw, obsh]
    radar_data = [radar_range, radar_angle]
   
    return obs_centroid, pickle_dir_and_prefix_cam, obs_size, pickle_dir_and_prefix_lidar, radar_data

 
# return predictions from lidar/camera 2d frontviews  
def predict_fcn(model, predict_file, dir_prefix, export, output_dir):  


    # load data
    predict_data = get_predict_data_matching_lidar_cam_frames(predict_file, dir_prefix)
    
    n_batches_per_epoch = data_number_of_batches_per_epoch(predict_data[1], BATCH_SIZE)
    
    # get some data
    predictions = model.predict_generator(
        data_generator_FCN(
                predict_data[0], predict_data[2], predict_data[1], predict_data[3],
                BATCH_SIZE, predict_data[4], None
            ),  # generator
        n_batches_per_epoch,
        verbose=0
    )

    #print predict_data[0]
    # reload data as one big batch
    all_data_loader = data_generator_FCN(
            predict_data[0], predict_data[2], predict_data[1], predict_data[3],
            len(predict_data[1]), predict_data[4], None
            )
    all_images, all_labels = all_data_loader.next()
    
    centroid_and_rz = np.zeros((all_images[0].shape[0],4))

    timestamps = []
    ind = 0

    # extract the 'car' category labels for all pixels in the first results, 0 is non-car, 1 is car
    for centroid, rz, file_prefix in zip(predictions[0], predictions[1], predict_data[1]):
    
        print centroid, rz

        centroid_and_rz[ind,0] = centroid[0]
        centroid_and_rz[ind,1] = centroid[1]
        centroid_and_rz[ind,2] = centroid[2]
        centroid_and_rz[ind,3] = rz[0]
        timestamps.append(os.path.basename(file_prefix).split('_')[0])
        ind += 1
    
    return centroid_and_rz, timestamps
    
        
def main():
    parser = argparse.ArgumentParser(description='Lidar car/pedestrian trainer')
    parser.add_argument("predict_file", type=str, default="", help="list of data folders for prediction or rosbag file name")
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--output_dir', type=str, default=None, help='output file for prediction results')
    parser.add_argument('--fcn_model', type=str, default="", help='Model Filename')
    parser.add_argument('--fcn_weights', type=str, default="", help='Weights Filename')
    
    
    
    parser.set_defaults(export=False)

    args = parser.parse_args()
    output_dir = args.output_dir
    predict_file = args.predict_file
    dir_prefix = args.dir_prefix
    prediction_file_name = "objects_obs1_camera_lidar_predictions.csv"

    fcn_net = load_fcn(args.fcn_model, args.fcn_weights, False, False, False)                
        
    xyz_pred, timestamps = predict_fcn(fcn_net, predict_file, dir_prefix, args.export, output_dir)        

    if output_dir is not None:
        file_prefix = output_dir + "/"

        write_prediction_data_to_csv(xyz_pred, timestamps, file_prefix + prediction_file_name)
        print('prediction result written to ' + file_prefix + prediction_file_name)    

        
if __name__ == '__main__':
    main()
