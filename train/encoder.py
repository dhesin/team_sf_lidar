# coding: utf-8
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import argparse
from math import sin, cos, sqrt
import json
import cv2
import csv
from process.globals import X_MIN, Y_MIN, Y_MAX, RES_RAD, CAM_IMG_TOP
from globals import IMG_CAM_WIDTH, IMG_CAM_HEIGHT, NUM_CAM_CHANNELS, NUM_REGRESSION_OUTPUTS
from keras.utils import to_categorical
from common.camera_model import CameraModel



print(Y_MIN, Y_MAX, RES_RAD)

def project_2d(tx, ty, tz):
    d = np.sqrt(tx ** 2 + ty ** 2)
    l2_norm = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)

    x_img = np.arctan2(-ty, tx) / RES_RAD[1]
    y_img = np.arcsin(tz/l2_norm) / RES_RAD[0]

    #print('tx={}, ty={}, tz={}, d={} l2={} arcsin={}'.format(tx,ty,tz,d, l2_norm, np.arcsin(tz/l2_norm)))

    # shift origin
    x_img -= X_MIN
    y_img -= Y_MIN

    y_img = int(y_img)
    x_img = int(x_img)

    y_img = min(y_img, Y_MAX)
    y_img = max(y_img, 0)

    y_img = int(Y_MAX - y_img)

    #return (y_img, x_img)
    return (x_img, y_img)

#returns the projected corners in order of distance from centroid in 2d
def get_bb(tx, ty, tz, rz, l, w, h):                   
    rot_z = np.array([[cos(rz), -sin(rz), 0.0], 
                      [sin(rz), cos(rz),  0.0],
                      [0.0,     0.0,      1.0]]) 
    
    bbox_3d = np.array([[tx-l/2., ty+w/2., tz+h/2.],
                     [tx-l/2., ty+w/2., tz-h/2.],
                     [tx-l/2., ty-w/2., tz+h/2.],
                     [tx-l/2., ty-w/2., tz-h/2.],
                     [tx+l/2., ty+w/2., tz+h/2.],
                     [tx+l/2., ty+w/2., tz-h/2.],
                     [tx+l/2., ty-w/2., tz+h/2.],
                     [tx+l/2., ty-w/2., tz-h/2.]])
    bbox_3d = (np.matmul(rot_z, bbox_3d.transpose())).transpose()
    
    bbox = np.zeros((8, 2), dtype='int')
    for i, rotated in enumerate(bbox_3d):           
        projected = project_2d(rotated[0], rotated[1], rotated[2])
        bbox[i, :] = [projected[0], projected[1]]
        
    centroid = project_2d(tx, ty, tz)
    d = []
    for p in bbox:
        d.append(distance(centroid, (p[0], p[1])))

    d = np.array(d)
    indices = np.argsort(d)

    sorted_corners = bbox[indices]
    return sorted_corners

#distance between two points in 2D
def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def area_from_corners(corner1, corner2):
    diff_x = abs(corner1[0] - corner2[0])
    diff_y = abs(corner1[1] - corner2[1])
    return diff_x * diff_y


def get_inner_rect(tx, ty, tz, rz, l, w, h):
    bbox = get_bb(tx, ty, tz, rz, l, w, h)
    sorted_corners = bbox[:4]

    upper_left_x = sorted_corners.min(axis=0)[0]
    upper_left_y = sorted_corners.min(axis=0)[1]
    lower_right_x = sorted_corners.max(axis=0)[0]
    lower_right_y = sorted_corners.max(axis=0)[1]
    return (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)


def get_outer_rect(tx, ty, tz, rz, l, w, h):
    bbox = get_bb(tx, ty, tz, rz, l, w, h)
    sorted_corners = bbox[-4:]

    upper_left_x = sorted_corners.min(axis=0)[0]
    upper_left_y = sorted_corners.min(axis=0)[1]
    lower_right_x = sorted_corners.max(axis=0)[0]
    lower_right_y = sorted_corners.max(axis=0)[1]
    return (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)


def get_circle_rect(tx, ty, tz, rz, l, w, h):
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_inner_rect(tx, ty, tz, rz, l, w, h)

    dim_x = (lower_right_x - upper_left_x)
    dim_y = (lower_right_y - upper_left_y)

    r = min(dim_y, dim_x)

    center_point_x = upper_left_x + dim_x / 2
    center_point_y = upper_left_y + dim_y / 2

    return (center_point_x - r / 2, center_point_y - r / 2), (center_point_x + r / 2, center_point_y + r / 2)

def generate_label_from_circle(tx, ty, tz, rz, l, w, h, INPUT_SHAPE):
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_circle_rect(tx, ty, tz, rz, l, w, h)
    r = min((lower_right_y - upper_left_y) / 2.0, (lower_right_x - upper_left_x) / 2.0)
    centroid = project_2d(tx, ty, tz)

    label = np.zeros(INPUT_SHAPE[:2])

    #print(upper_left_x, lower_right_x)
    #print(upper_left_y, lower_right_y)

    for x in range(upper_left_x, lower_right_x, 1):
        for y in range(upper_left_y, lower_right_y, 1):
            if distance(centroid, (x, y)) <= r:
                label[y, x] = 1

    # label[upper_left_x:lower_right_x, upper_left_y:lower_right_y] = 0
    y = to_categorical(label, num_classes=2)  # 1st dimension: on-vehicle, 2nd dimension: off-vehicle

    return y


def get_label_bounds(tx, ty, tz, rz, l, w, h, method='outer_rect'):
    if method == 'circle':
        return get_circle_rect(tx, ty, tz, rz, l, w, h)
    else:
        if method == 'inner_rect':
            return get_inner_rect(tx, ty, tz, rz, l, w, h)
        elif method == 'outer_rect':
            return get_outer_rect(tx, ty, tz, rz, l, w, h)
    return None


def generate_label(tx, ty, tz, rx, ry, rz, l, w, h, INPUT_SHAPE, method='outer_rect', image=None):
    if method == 'circle':
        (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_circle_rect(tx, ty, tz, l, w, h)
        y = generate_label_from_circle(tx, ty, tz, rz, l, w, h, INPUT_SHAPE, rz)
    else:
        if method == 'inner_rect':
            (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_inner_rect(tx, ty, tz, rz, l, w, h)
        elif method == 'outer_rect':
            (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_outer_rect(tx, ty, tz, rz, l, w, h)
        #print (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)

        label = np.zeros(INPUT_SHAPE[:2])
        label[upper_left_y:lower_right_y, upper_left_x:lower_right_x] = 1
        y = to_categorical(label, num_classes=2) #1st dimension: on-vehicle, 2nd dimension: off-vehicle
        y = y.astype('float')

    # groud truths for regression part.. encode bounding box corners in 3D
    rot_z = np.array([[cos(rz), -sin(rz), 0.0], 
                      [sin(rz), cos(rz),  0.0],
                      [0.0,             0.0,        1.0]]) 
    
    bbox = np.array([[tx-l/2., ty+w/2., tz+h/2.],
                     [tx-l/2., ty+w/2., tz-h/2.],
                     [tx-l/2., ty-w/2., tz+h/2.],
                     [tx-l/2., ty-w/2., tz-h/2.],
                     [tx+l/2., ty+w/2., tz+h/2.],
                     [tx+l/2., ty+w/2., tz-h/2.],
                     [tx+l/2., ty-w/2., tz+h/2.],
                     [tx+l/2., ty-w/2., tz-h/2.]])
    bbox = (np.matmul(rot_z, bbox.transpose())).transpose()
    #bbox.append(((tx-l/2.), ty+w/2., tz+h/2.))
    #bbox.append((tx-l/2., ty+w/2., tz-h/2.))
    #bbox.append((tx-l/2., ty-w/2., tz+h/2.))
    #bbox.append((tx-l/2., ty-w/2., tz-h/2.))
    #bbox.append((tx+l/2., ty+w/2., tz+h/2.))
    #bbox.append((tx+l/2., ty+w/2., tz-h/2.))
    #bbox.append((tx+l/2., ty-w/2., tz+h/2.))
    #bbox.append((tx+l/2., ty-w/2., tz-h/2.))
    
    
    gt_regression = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_REGRESSION_OUTPUTS), dtype='float')    
    
    if image is None:
        for ind, values in enumerate(bbox):
            gt_regression[:, :, 3*ind] = values[0]*label[:,:]
            gt_regression[:, :, 3*ind+1] = values[1]*label[:,:]
            gt_regression[:, :, 3*ind+2] = values[2]*label[:,:]
    else:
        c = np.array(bbox)         
        
        for img_x in range(upper_left_x, lower_right_x):
            for img_y in range(upper_left_y, lower_right_y):
                distance = image[img_y, img_x, 0]
                height = image[img_y, img_x, 1]
                theta = (img_x + X_MIN) * RES_RAD[1]
                phi = (img_y + Y_MIN) * RES_RAD[0] 
                px = distance * cos(theta)
                py = - distance * sin(theta)
                pz = height
                p = np.array([px, py, pz])
                
                #rotation around z axis                                                     
                rot_z = np.array([[cos(theta), -sin(theta), 0.0], 
                                  [sin(theta), cos(theta),  0.0],
                                  [0.0,             0.0,              1.0]])  
                                
                #rotation around y axis  
                rot_y = np.array([[cos(phi), 0.0, sin(phi)],
                                  [0.0,           1.0, 0.0],
                                  [-sin(phi),0.0, cos(phi)]])
                               
                rot = np.matmul(rot_z, rot_y)
                rot_T = rot.transpose()
                                
                c_prime = np.matmul(rot_T, (c - p).transpose()) 
                c_prime_T = c_prime.transpose()                                
                gt_regression[img_y, img_x, :] = np.reshape(c_prime_T, (-1))
              
    gt_regression = np.reshape(gt_regression, (INPUT_SHAPE[0]*INPUT_SHAPE[1], NUM_REGRESSION_OUTPUTS))
    
    labels_concat = np.concatenate((y, gt_regression), axis=1) 
    #return y
    return labels_concat

def generate_camera_bb(tx, ty, tz, l, w, h, camera_model):

    bbox = []    
    bbox.append([tx-l/2., ty+w/2., tz+h/2., 1.])
    bbox.append([tx-l/2., ty+w/2., tz-h/2., 1.])
    bbox.append([tx-l/2., ty-w/2., tz+h/2., 1.])
    bbox.append([tx-l/2., ty-w/2., tz-h/2., 1.])
    bbox.append([tx+l/2., ty+w/2., tz+h/2., 1.])
    bbox.append([tx+l/2., ty+w/2., tz-h/2., 1.])
    bbox.append([tx+l/2., ty-w/2., tz+h/2., 1.])
    bbox.append([tx+l/2., ty-w/2., tz-h/2., 1.])
    uv_bbox = camera_model.project_lidar_points_to_camera_2d(bbox)
    uv_bbox = np.asarray(uv_bbox, dtype='int')
 
    centroid = []
    centroid.append([tx, ty, tz, 1.])
    uv_centroid = camera_model.project_lidar_points_to_camera_2d(centroid)
    uv_centroid = np.asarray(uv_centroid, dtype='int')
    
    d = []
    for p in uv_bbox:
        d.append(distance(uv_centroid[0], (p[0], p[1])))

    d = np.asarray(d, dtype='int')
    indices = np.argsort(d)
    sorted_corners = uv_bbox[indices]
    sorted_corners[:,1] = sorted_corners[:,1] - CAM_IMG_TOP
    uv_centroid[:,1] = uv_centroid[:,1] - CAM_IMG_TOP
    return sorted_corners, uv_centroid
    
def generate_camera_label(tx, ty, tz, l, w, h, INPUT_SHAPE, camera_model, method='outer_rect'):

    uv_bbox_sorted, uv_centroid = generate_camera_bb(tx, ty, tz, l, w, h, camera_model)
    r = 0
        
    if method == 'circle':
        uv_bbox = uv_bbox_sorted[:4]
        upper_left_y = uv_bbox.min(axis=0)[1]
        upper_left_x = uv_bbox.min(axis=0)[0] 
        lower_right_y = uv_bbox.max(axis=0)[1]
        lower_right_x = uv_bbox.max(axis=0)[0] 
        width = (lower_right_x - upper_left_x)
        height = (lower_right_y - upper_left_y)
        
        r = min(width, height)
        center_point_x = upper_left_x + width / 2
        center_point_y = upper_left_y + height / 2
        
        upper_left_x = center_point_x - r / 2
        upper_left_y = center_point_y - r / 2
        lower_right_x = center_point_x + r / 2
        lower_right_y = center_point_y + r / 2
        
        label = np.zeros(INPUT_SHAPE[:2])
        for x in range(upper_left_x, lower_right_x, 1):
            for y in range(upper_left_y, lower_right_y, 1):
                if distance((uv_centroid[0][0], uv_centroid[0][1]), (x, y)) <= r:
                    label[y, x] = 1
         #print (upper_left_y, lower_right_y), (upper_left_x, lower_right_x)
    else:      
        if method == 'inner_rect':
            uv_bbox = uv_bbox_sorted[:4]     
        elif method == 'outer_rect':
            uv_bbox = uv_bbox_sorted[-4:]   
                              
        upper_left_y = uv_bbox.min(axis=0)[1]
        upper_left_x = uv_bbox.min(axis=0)[0] 
        lower_right_y = uv_bbox.max(axis=0)[1]
        lower_right_x = uv_bbox.max(axis=0)[0] 
        width = (lower_right_x - upper_left_x)
        height = (lower_right_y - upper_left_y)
        
        x_margin = width/4
        y_margin = height/4
        #print x_margin, y_margin
        
        upper_left_y -= y_margin
        upper_left_x -= x_margin
        lower_right_y += y_margin
        lower_right_x += x_margin
        
        label = np.zeros(INPUT_SHAPE[:2])
        label[upper_left_y:lower_right_y, upper_left_x:lower_right_x] = 1

    y = to_categorical(label, num_classes=2) #1st dimension: on-vehicle, 2nd dimension: off-vehicle  
    #print np.sum(y[:,0]) , np.sum(y[:,1]), y.shape, np.sum(y[:,0])+np.sum(y[:,1])
    
    return y, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), uv_bbox_sorted, uv_centroid, r

    
def draw_bb_circle(tx, ty, tz, rz, l, w, h, infile, outfile):
    centroid = project_2d(tx, ty, tz)
    #print('Centroid: {}'.format(centroid))
    bbox = get_bb(tx, ty, tz, rz, l, w, h)
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_inner_rect(tx, ty, tz, rz, l, w, h)
    #print (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)
    r = min((lower_right_y - upper_left_y)/2.0, (lower_right_x - upper_left_x)/2.0)
    #print r

    img = cv2.imread(infile)
    cv2.circle(img, centroid, 2, (0, 0, 255), thickness=-1)

    for p in bbox[:4]:
        cv2.circle(img, (p[0], p[1]), 2, (255, 255, 255), thickness=-1)
    for p in bbox[-4:]:
        cv2.circle(img, (p[0], p[1]), 2, (0, 255, 0), thickness=-1)

    cv2.circle(img, centroid, int(r), (0, 255, 255), thickness=2)

    #save image
    cv2.imwrite(outfile, img)


def draw_bb_rect(tx, ty, tz, rz, l, w, h, infile, outfile, method='inner_rect'):
    centroid = project_2d(tx, ty, tz)
    img = cv2.imread(infile)
    cv2.circle(img, centroid, 2, (0, 0, 255), thickness=-1)

    bbox = get_bb(tx, ty, tz, rz, l, w, h)
    for p in bbox:
        cv2.circle(img, (p[0], p[1]), 2, (255, 255, 255), thickness=-1)

    if method == 'inner_rect':
        upper_left, lower_right = get_inner_rect(tx, ty, tz, rz, l, w, h)
    elif method == 'outer_rect':
        upper_left, lower_right = get_outer_rect(tx, ty, tz, rz, l, w, h)

    cv2.rectangle(img, upper_left, lower_right, (0, 255, 0), 1)

    #save image
    cv2.imwrite(outfile, img)


def draw_bb(tx, ty, tz, rz, l, w, h, infile, outfile, method='circle'):
    if method == 'circle':
        draw_bb_circle(tx, ty, tz, rz, l, w, h, infile, outfile)
    else:
        draw_bb_rect(tx, ty, tz, rz, l, w, h, infile, outfile, method)


def test():
    #gps_l, gps_w, gps_h = (2.032, 0.7239, 1.6256)
    l, w, h = (4.2418,1.4478,1.5748)

    #centroid of obstacle after interpolation
    #tx, ty, tz = (0.699597401296,-76.989,2.17780519741) #old 10.bag
    tx, ty, tz = (-0.8927325054898647, -3.6247593094278256, -0.648832347271497) #10.bag
    #tx, ty, tz = (-6.81401019142,-84.618,2.0329898085) #old 4_f.bag
    #tx, ty, tz = (9.083115901203417, 0.04826503520317882, -0.47151975040470145) #4_f.bag
    draw_bb(tx, ty, tz, l, w, h, '../sample/10/out/lidar_360/1490991699437114271_distance.png', '../sample/10_1490991699437114271_distance_circle.png', 'circle')
    draw_bb(tx, ty, tz, l, w, h, '../sample/10/out/lidar_360/1490991699437114271_distance.png', '../sample/10_1490991699437114271_distance_inner.png', 'inner_rect')
    draw_bb(tx, ty, tz, l, w, h, '../sample/10/out/lidar_360/1490991699437114271_distance.png', '../sample/10_1490991699437114271_distance_outer.png', 'outer_rect')
    y = generate_label(tx, ty, tz, l, w, h, (32, 1801, 3), method='circle')
    print(np.nonzero(y[:,0])[0].shape[0])
    
def main():
    parser = argparse.ArgumentParser(description="Draw bounding box on projected 2D lidar images.")
    parser.add_argument("--input_dir", help="Input directory.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--shape", help="bounding box shape: circle, outer_rect, inner_rect", default="outer_rect")
    parser.add_argument('--data_source', type=str, default="lidar", help='lidar or camera data')
    parser.add_argument('--camera_model', type=str, help='Camera calibration yaml')
    parser.add_argument('--lidar2cam_model', type=str, help='Lidar to Camera calibration yaml')
    parser.add_argument('--metadata', type=str, help='path/filename to metadata.csv')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    shape = args.shape
    data_source = args.data_source
    camera_model_file = args.camera_model
    lidar2cam_model_file = args.lidar2cam_model
    metadata_fname = args.metadata
    print(shape)
    
    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        print('input_dir or output_dir does not exist')
        sys.exit()

    if shape not in ('circle', 'outer_rect', 'inner_rect'):
        print('shape must be one of the following: circle, outer_rect, inner_rect')
        sys.exit()
    
    if not metadata_fname:
        print "need to enter metadata.csv file path/name"
        exit(1) 
            
    with open(metadata_fname) as metafile:
        records = csv.DictReader(metafile)
        mdr = []
        for record in records:
            mdr.append(record)
        
        print mdr[0]
        l = float(mdr[0]['l'])
        w = float(mdr[0]['w'])
        h = float(mdr[0]['h'])
            
    if data_source == "lidar":
        #input_dir needs to contain the following:
        #obs_poses_interp_transform.csv, and a sub directory lidar_360 that contains lidar images    
        obs_file = os.path.join(input_dir, 'obs_poses_interp_transform.csv')
        if not os.path.exists(obs_file):
            print('missing obs_poses_interp_transform.csv')
            sys.exit()
            
        obs_df = pd.read_csv(obs_file, index_col=['timestamp'])    
        #print(obs_df)

        lidar_img_dir = os.listdir(os.path.join(input_dir, 'lidar_360'))       

        for f in lidar_img_dir:
            if f.endswith('_distance.png'):
                ts = int(f.split('_')[0])

                if ts in list(obs_df.index):
                    tx = obs_df.loc[ts]['tx']
                    ty = obs_df.loc[ts]['ty']
                    tz = obs_df.loc[ts]['tz']
                    rz = obs_df.loc[ts]['rz']
                    infile = os.path.join(input_dir, 'lidar_360', f)
                    outfile = os.path.join(output_dir, f.split(".")[0] + '_bb.png')
                    draw_bb(tx, ty, tz, rz, l, w, h, infile, outfile, method=shape)           

    elif data_source == "camera":
       if not camera_model_file:
            print "need to enter camera calibration yaml"
            exit(1)
       if not lidar2cam_model_file:
            print "need to enter lidar to camera calibration yaml"
            exit(1)                                       
                    
       image_width = IMG_CAM_WIDTH
       image_height = IMG_CAM_HEIGHT
       input_shape = (IMG_CAM_HEIGHT, IMG_CAM_WIDTH, NUM_CAM_CHANNELS)  
       num_channels = NUM_CAM_CHANNELS
        
       camera_model = CameraModel()
       camera_model.load_camera_calibration(camera_model_file, lidar2cam_model_file)
       
       obs_file = os.path.join(input_dir, 'obs_poses_camera.csv')
       if not os.path.exists(obs_file):
            print('missing obs_poses_camera.csv')
            sys.exit()
            
       obs_df_cam = pd.read_csv(obs_file, index_col=['timestamp'])    
       #print(obs_df_cam)


       cam_img_dir = os.listdir(os.path.join(input_dir, 'camera'))

       for f in cam_img_dir:
            if f.endswith('_image.png'):
                ts = int(f.split('_')[0])

                if ts in list(obs_df_cam.index):
                    tx = obs_df_cam.loc[ts]['tx']
                    ty = obs_df_cam.loc[ts]['ty']
                    tz = obs_df_cam.loc[ts]['tz']
                    infile = os.path.join(input_dir, 'camera', f)
                    outfile = os.path.join(output_dir, f.split(".")[0] + '_cam_bb.png')
                    y, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), uv_bbox, uv_centroid, r = \
                            generate_camera_label(tx, ty, tz, l, w, h, (image_height, image_width), camera_model, shape)
                    
                    img = cv2.imread(infile)
                    #if 0 < upper_left_x < image_width and 0< upper_left_y < image_height and  \
                    #    0 < lower_right_x < image_width and 0< lower_right_y < image_height:
                        #print img.shape
                        #print tx, ty, tz
                        #print (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)
                        #print y.shape
                    for p in uv_bbox:
                        cv2.circle(img, (p[0], p[1]), 2, (0, 0, 255), thickness=-1)
                    cv2.circle(img, (uv_centroid[0][0], uv_centroid[0][1]), 2, (255, 0, 255), thickness=-1) 
                     
                    if shape == "circle":  
                        cv2.circle(img, (uv_centroid[0][0], uv_centroid[0][1]), int(r), (0, 255, 255), thickness=2) 
                    else:  
                        cv2.rectangle(img, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), (0, 255, 0), 5)
                    
                    
                    cv2.imwrite(outfile, img)
                    #y = y*255
                    #cv2.imwrite(outfile + "_label.jpg",(np.reshape(y, (image_height, image_width, 2)))[:,:,1])


if __name__ == '__main__':
    main()
