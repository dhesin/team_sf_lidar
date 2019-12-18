#!/usr/bin/python
"""Rectify images under given path according to the camera_matrix and distortion_coefficients in provided yaml file
   usage: python rectify_image.py <YAML_PATH> <IMG_DIR> <OUTPUT_DIR>
"""

import yaml
import rosbag
import sensor_msgs.msg
import os
import sys
import cv2
import numpy as np
import argparse

def extract_calib_info(yaml_path):
    try:
        yaml_stream = file(yaml_path, 'r')
        yaml_data = yaml.load(yaml_stream)
        calibration_info = sensor_msgs.msg.CameraInfo()
        calibration_info.K = yaml_data['camera_matrix']['data']
        calibration_info.D = yaml_data['distortion_coefficients']['data']
        calibration_info.R = yaml_data['rectification_matrix']['data']
        calibration_info.P = yaml_data['projection_matrix']['data']
        calibration_info.width = yaml_data['image_width']
        calibration_info.height = yaml_data['image_height']
        calibration_info.distortion_model = yaml_data['distortion_model']
    except Exception as e:
        import traceback
        traceback.print_exc()
    return calibration_info


def initUndistortRectifyMap(calibration_info):
    img_dim = (calibration_info.width,calibration_info.height)
    mtx = (np.array(calibration_info.K)).reshape(3, 3)
    dist = np.array(calibration_info.D)
    rectif = np.array(calibration_info.R).reshape(3, 3)
    #print mtx, dist
    newcameramtx, new_size = cv2.getOptimalNewCameraMatrix(mtx, dist, img_dim, 1, img_dim)
    mapx,mapy = cv2.initUndistortRectifyMap(mtx, dist, rectif, newcameramtx, img_dim, 5)
    return mapx, mapy, new_size


def remap(img,mapx,mapy,new_size):
    rectified = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    x,y,w,h = new_size
    rectified = rectified[y:y+h, x:x+w]
    return rectified
    
    
def main():
    parser = argparse.ArgumentParser(description="Rectify images under given path according to the camera_matrix and distortion_coefficients in provided yaml file")
    parser.add_argument("yaml_path", help="Path to yaml file containing the camera_matrix and distortion_coefficients")
    parser.add_argument("img_dir", help="Input directory")
    parser.add_argument("output_dir", help="Output directory")

    args = parser.parse_args()
    yaml_path = args.yaml_path
    img_dir = args.img_dir
    output_dir = args.output_dir
    

    if not os.path.isfile(yaml_path):
        print('yaml_path ' + yaml_path + ' does not exist')
        sys.exit()

    if not os.path.isdir(img_dir):
        print('img_dir ' + img_dir + ' does not exist')
        sys.exit()

    if not os.path.isdir(output_dir):
        print('output_dir ' + output_dir + ' does not exist')
        sys.exit()
    
    calibration_info = extract_calib_info(yaml_path)
    mapx, mapy, new_size = initUndistortRectifyMap(calibration_info)
    for filename in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir,filename))
        if img is not None:
            rectified = remap(img,mapx,mapy,new_size)
            cv2.imwrite(os.path.join(output_dir,filename), rectified)
                

if __name__ == '__main__':
    main()