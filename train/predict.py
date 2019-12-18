import sys
import os
import argparse
import json
import datetime
import numpy as np
import cv2
from math import sin, cos, atan2, pi
import csv
import rosbag
import sensor_msgs.point_cloud2
print(sys.path)
sys.path.append('../')
from common.camera_model import CameraModel
from process.globals import X_MIN, Y_MIN, RES, RES_RAD, LIDAR_MIN_HEIGHT
from scipy.ndimage.measurements import label
from globals import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, \
                    INPUT_SHAPE, BATCH_SIZE, \
                    PREDICTION_FILE_NAME, PREDICTION_MD_FILE_NAME, \
                    IMG_CAM_WIDTH, IMG_CAM_HEIGHT, NUM_CAM_CHANNELS
from loader import get_data, data_number_of_batches_per_epoch, data_generator_train, data_generator_predict
import model as model_module
from process.extract_rosbag_lidar import generate_lidar_2d_front_view
from encoder import distance, project_2d

from keras.models import model_from_json

MIN_PROB = 0.5 #lowest probability to participate in heatmap 
MIN_BBOX_AREA = 100 #minimum area of the bounding box to be declared a car
MIN_HEAT = 2 #lowest number of heat allowed
MAX_BBX_DIST = 5. #maximum distance for neighbouring bounding boxes to be clustered

def find_obstacle(y_pred, input_shape):
    y_label = y_pred[:,1].flatten()      
    pred = np.reshape(y_label, input_shape[:2])
    ones = np.where(pred >= MIN_PROB)
    
    # Generate bounding box of size 4 for each positive pixel
    bbox_list = []
    for i in range(len(ones[0])):
        y = ones[0][i]
        x = ones[1][i]
        bbox = ((y - 2, x - 2), (y + 2, x + 2))
        bbox_list.append(bbox)
     
    heatmap = np.zeros_like(pred).astype(np.float)
    # Add heat to each box in box list
    for box in bbox_list:
        heatmap[box[0][0]:box[1][0], box[0][1]:box[1][1]] += 1
            
    # Apply threshold to help remove false positives
    heatmap[heatmap <= MIN_HEAT] = 0
    labels = label(heatmap)    
    #print(labels)

    # Iterate through all detected clusters
    max_area = 0
    for cluster_number in range(1, labels[1]+1):
        # Find pixels with each label value
        nonzero = (labels[0] == cluster_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        #print(bbox)
        width = np.max(nonzerox) - np.min(nonzerox)
        height = np.max(nonzeroy) - np.min(nonzeroy)
        area = width * height
        if area > max_area:
            largest_bbox = bbox
            max_area = area
    
    #print('max_area={}'.format(max_area))
    if max_area <= MIN_BBOX_AREA:
        return None, None, None
    
    largest_bbox = ((largest_bbox[0][0] + 2, largest_bbox[0][1] + 2), (largest_bbox[1][0] - 2, largest_bbox[1][1] - 2))
    centroid_x = int((largest_bbox[0][0] + largest_bbox[1][0]) / 2.0)
    centroid_y = int((largest_bbox[0][1] + largest_bbox[1][1]) / 2.0)
    
    return (centroid_x, centroid_y), largest_bbox, max_area

def is_far(corners, centroid_3d):
    #delta = [5.0, 2.0, 2.0]
    delta = [9.0, 3.0, 3.0]
    for c in corners:
        if (c[0] > centroid_3d[0] + delta[0] or c[0] < centroid_3d[0] - delta[0]) \
               or (c[1] > centroid_3d[1] + delta[1] or c[1] < centroid_3d[1] - delta[1]) \
               or (c[2] > centroid_3d[2] + delta[2] or c[2] < centroid_3d[2] - delta[2]):
            return True
       
    return False
    
def find_bbox_3d(distance_img, height_img, y_pred, bbox_2d, centroid_3d):
    y_pred = np.reshape(y_pred, (distance_img.shape[0], distance_img.shape[1], 26))    
    pos = np.where(y_pred[:, :, 1] >= MIN_PROB)   
    pos_pixels = y_pred[pos]
        
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = bbox_2d    
    bbox = []
    # revert prediction encoding to uncover bbox
    # limit to the 2d positive pixels to improve processing time
    for img_x in range(upper_left_x - 100, lower_right_x + 100):
        for img_y in range(upper_left_y - 2, lower_right_y + 2):
    #for img_y in pos[0]:
    #    for img_x in pos[1]:
            if not (img_x in pos[1] and img_y in pos[0]):
                continue
                
            distance = distance_img[img_y, img_x]           
            height = height_img[img_y, img_x]
            theta = (img_x + X_MIN) * RES_RAD[1]
            phi = (img_y + Y_MIN) * RES_RAD[0] 
            px = distance * cos(theta)
            py = - distance * sin(theta)
            pz = height
            p = np.array([px, py, pz])
            c_prime = np.reshape(y_pred[img_y, img_x, 2:], (8, 3))
            
            #rotation around z axis                                                     
            rot_z = np.array([[cos(theta), -sin(theta), 0.0], 
                              [sin(theta), cos(theta),  0.0],
                              [0.0,             0.0,              1.0]])  
                            
            #rotation around y axis  
            rot_y = np.array([[cos(phi), 0.0, sin(phi)],
                              [0.0,           1.0, 0.0],
                              [-sin(phi),0.0, cos(phi)]])
                           
            rot = np.matmul(rot_z, rot_y)            
            c = (np.matmul(rot, c_prime.transpose())).transpose() + p  #shape: (8, 3)
             
            # limit to boxes that are not too far away from the centroid (to improve processing time)
            if not is_far(c, centroid_3d):            
                bbox.append(c)
   
    if len(bbox) == 0:
        return np.zeros((7)), None
        
    bbox = np.array(bbox)
    candidate_bbox = np.zeros((8, 3))  
    
    # the bbox that has the most neighbouring boxes within a small distance is the candidate    
    min_count = 0        
    counts = []
    for c in bbox:
        count = 0        
            
        for c2 in bbox:                    
            d = np.linalg.norm(c - c2)             
            if d > 0 and d < MAX_BBX_DIST:                    
                count += 1        
        
        if count > min_count:
            min_count = count           
            candidate_bbox = c
        counts.append(count) 
    
    # there may be ties on the max count. Use all.
    counts = np.array(counts)
    indices = np.where(counts == counts.max())    
    bbox = bbox[indices]          
    
    candidate_bbox = bbox.mean(axis=0)   
    #print('candidate_bbox: {}'.format(candidate_bbox))
    pred = np.zeros((7))
    pred[:3] = np.mean(candidate_bbox, 0)
   
    # estimate obstacle orientation and size
    yaws = []
    box_l_array = []
    box_w_array = []
    box_h_array = []
    for i in range(4):
        # 1st point in the front, 2nd point in the back
        delta_x = candidate_bbox[i, 0] - candidate_bbox[i+4, 0]
        delta_y = candidate_bbox[i, 1] - candidate_bbox[i+4, 1]
        yaw = atan2(delta_y, delta_x)
        yaws.append(yaw)
        
        box_l = delta_x / cos(yaw) if yaw != pi/2.0 else delta_y 
        box_l_array.append(abs(box_l))
        
        # 1st point on the left, 2nd point on the right
        delta_x = candidate_bbox[i, 0] - candidate_bbox[i+2, 0]
        delta_y = candidate_bbox[i, 1] - candidate_bbox[i+2, 1]
        box_w = delta_y / cos(yaw) if yaw != pi/2.0 else delta_x
        box_w_array.append(abs(box_w))
        
        # 1st point on the top, 2nd point on the bottom
        delta_z = candidate_bbox[i, 2] - candidate_bbox[i+1, 2]
        box_h_array.append(abs(delta_z))    
    
    pred[3] = np.mean(yaws)  
    pred[4] = np.mean(box_l_array)
    pred[5] = np.mean(box_w_array)
    pred[6] = np.mean(box_h_array)               
        
    return pred, candidate_bbox
    
 # project the 3d centroid and bbox to 2d, for visualization purpose. Can be removed if not needed
def project_bbox_2d(centroid_3d, bbox_3d):
    centroid_2d = project_2d(centroid_3d[0],centroid_3d[1], centroid_3d[2])
    bboxes_2d = []    
    d = []
       
    for c in bbox_3d:        
        x, y = project_2d(c[0], c[1], c[2])   
        bboxes_2d.append((x, y))
        d.append(distance(centroid_2d, (x, y)))
    
    bboxes_2d = np.array(bboxes_2d)
    d = np.array(d)
    indices = np.argsort(d)
    sorted_corners = bboxes_2d[indices]
    #sorted_corners = sorted_corners[4:]

    upper_left_x = sorted_corners.min(axis=0)[0]
    upper_left_y = sorted_corners.min(axis=0)[1]
    lower_right_x = sorted_corners.max(axis=0)[0]
    lower_right_y = sorted_corners.max(axis=0)[1]
    bbox_2d = (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)
    
    return centroid_2d, bbox_2d
            
#
# take in find_obstacle() batch output (centroid x/theata,y/phi) and batch input(distance map) 
# and back project 2D centroid to 3D centroid(x,y,z)
#
def back_project_2D_2_3D(centroids, bboxes, distance_data, height_data):       
    xyz_coor = np.zeros((centroids.shape[0], 4))
    
    w = distance_data[0,:,:].shape[1]
    h = distance_data[0,:,:].shape[0]    
    
    valid_points_mask = np.logical_and(distance_data > 0, height_data > LIDAR_MIN_HEIGHT)
    
    ind_y, ind_x = np.unravel_index(np.arange(w*h),(h,w))
    ind_y_2d = np.reshape(ind_y,(h,w))
    ind_x_2d = np.reshape(ind_x,(h,w))        
        
    for i in range(centroids.shape[0]):         
        if (not(valid_points_mask[i,int(centroids[i,1]), int(centroids[i,0])]) and (bboxes[i,0] != 0) and (bboxes[i,2] != 0)):
        
            bb_left = int(bboxes[i,0])
            bb_right = int(bboxes[i,2])+1
            bb_top = int(bboxes[i,1])
            bb_bottom = int(bboxes[i,3])+1  
            #print('bounding box left {} right {} top {} bottom {}'.format(bb_left, bb_right, bb_top, bb_bottom))                           
                                    
            dist_x = ind_x_2d[bb_top:bb_bottom,bb_left:bb_right] - int(centroids[i,0])
            dist_y = ind_y_2d[bb_top:bb_bottom,bb_left:bb_right] - int(centroids[i,1])
            dist_2_centroid = np.sqrt(dist_x*dist_x + dist_y*dist_y)
            
            dist_2_centroid_valid = np.where(valid_points_mask[i, bb_top:bb_bottom,bb_left:bb_right], dist_2_centroid, 10e7)
            min_ind = np.argmin(dist_2_centroid_valid)
            min_val = np.min(dist_2_centroid_valid)   
            
            #print('min index {} min value {}'.format(min_ind, min_val))
            
            # cannot find any valid point.. zero out centroid and bounding box
            if (min_val == 10e7):
                centroids[i,1] = 0
                centroids[i,0] = 0  
                bboxes[i,0] = 0
                bboxes[i,1] = 0
                bboxes[i,2] = 0
                bboxes[i,3] = 0
                print('cannot find valid centroid')
            else:
                new_ind_1, new_ind_0 = np.unravel_index([min_ind],(bb_bottom-bb_top,bb_right-bb_left))
                new_ind_1 += bb_top
                new_ind_0 += bb_left
                #print(' new centroid selected old: {} {} new: {} {}'.format(centroids[i,1], centroids[i,0], new_ind_1, new_ind_0))
                centroids[i,1], centroids[i,0] = new_ind_1, new_ind_0
        
        if not np.array_equal(centroids[i,:], [0, 0]):
            distance = distance_data[i, int(centroids[i,1]), int(centroids[i,0])]
            height = height_data[i, int(centroids[i,1]), int(centroids[i,0])]
            theata = (centroids[i,0] + X_MIN) * RES_RAD[1]
                  
            # increase  to approximate centroid - not surface of car
            distance += 0.75
            
            xyz_coor[i,0] = distance * cos(theata)
            xyz_coor[i,1] = - distance * sin(theata)
            xyz_coor[i,2] = height
        
        #print('centroid: {}, height: {}, theta: {}, x: {}, y: {}, z: {}'.
        #     format(centroids[i], height, theata,
        #            xyz_coor[i,0], xyz_coor[i,1], xyz_coor[i,2]))

    return xyz_coor


def write_prediction_data_to_csv(pred_result, timestamps, output_file):

    csv_file = open(output_file, 'w')
    writer = csv.DictWriter(csv_file, ['timestamp', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'l', 'w', 'h'])

    writer.writeheader()

    for pred, ts in zip(pred_result, timestamps):   
        if pred.shape[0] == 3:
            writer.writerow({'timestamp': ts, 'tx': pred[0], 'ty': pred[1], 'tz': pred[2],
                        'rx': 0.0, 'ry': 0.0, 'rz': 0.0,
                        'l': 0.0, 'w': 0.0, 'h': 0.0})
        else:
            writer.writerow({'timestamp': ts, 'tx': pred[0], 'ty': pred[1], 'tz': pred[2],
                        'rx': 0.0, 'ry': 0.0, 'rz': pred[3],
                        'l': pred[4], 'w': pred[5], 'h': pred[6]})

def write_metadata(pred_result, output_file):
    with open(output_file, 'w') as out:
        out.write('l,w,h\n')
        pred_result = pred_result[:, 4:]    
        pred_result = pred_result[~np.all(pred_result == 0.0, axis=1)]
        #print('estimated obs size: {}'.format(pred_result))
        size = np.mean(pred_result, axis=0)        
        out.write('{:.4f},{:.4f}, {:.4f}\n'.format(size[0], size[1], size[2]))       
        
def load_model(modelFile, weightsFile, use_regression):
    defaultWeightsFile = modelFile.replace('json', 'h5')
    if weightsFile != "":
        defaultWeightsFile = weightsFile

    model = model_module.load_model(modelFile, defaultWeightsFile,
                       INPUT_SHAPE, NUM_CLASSES, use_regression)
    model.trainable = False
    model.load_weights(defaultWeightsFile)
    
    return model
    
# return prection from a numpy array of point cloud        
def predict_point_cloud(model, points, cmap=None, use_regression=True):
    points_2d = generate_lidar_2d_front_view(points, cmap=cmap)    
            
    input = np.ndarray(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=float)
    input[:,:,0] = points_2d['distance_float']
    input[:,:,1] = points_2d['height_float']
    input[:,:,2] = points_2d['intensity_float']

    model_input = np.asarray([input])

    prediction = model.predict(model_input)
    
    centroid_2d, bbox_2d, _ = find_obstacle(prediction[0], INPUT_SHAPE)
    pred = np.array([0, 0, 0, 0])    
    
    if centroid_2d is not None:        
        centroids = np.array(centroid_2d).reshape(1, 2)
        bboxes = np.array(bbox_2d).reshape(1,4)
        distance_data = np.array(points_2d['distance_float']).reshape(1, IMG_HEIGHT, IMG_WIDTH)
        height_data = np.array(points_2d['height_float']).reshape(1, IMG_HEIGHT, IMG_WIDTH)
        centroid_3d = back_project_2D_2_3D(centroids, bboxes, distance_data, height_data)[0]
        
        if use_regression:
            if not (centroid_3d[0] == 0. and centroid_3d[1] == 0.):  
               pred, bbox_3d = find_bbox_3d(distance_data[0], height_data[0], prediction[0], bbox_2d, centroid_3d) 
        else:
            pred = centroid_3d[0]
    #print('predicted centroid: {}'.format(centroid_3d))
    
    return pred

# return predictions from rosbag
def predict_rosbag(model, predict_file, use_regression=True):
    bag = rosbag.Bag(predict_file, "r")        
    xyz_pred = []
    timestamps = []
    
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=False)
        points = np.array(list(points))
                
        pred = predict_point_cloud(model, points, use_regression=use_regression)
        xyz_pred.append(pred)
        timestamps.append(t)
    
    return xyz_pred, timestamps         
 
# return predictions from lidar/camera 2d frontviews  
def predict(model, predict_file, dir_prefix, export, output_dir, data_source, camera_model=None):  

    image_width = None
    image_height = None
    input_shape = None
    num_channels = None
    if data_source == "camera":
       image_width = IMG_CAM_WIDTH
       image_height = IMG_CAM_HEIGHT
       input_shape = (IMG_CAM_HEIGHT, IMG_CAM_WIDTH, NUM_CAM_CHANNELS)  
       num_channels = NUM_CAM_CHANNELS       
    elif data_source == "lidar":
        camera_model = None
        image_width = IMG_WIDTH
        image_height = IMG_HEIGHT
        input_shape = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
        num_channels= NUM_CHANNELS        
    else:
        print "invalid data source type"
        exit(1)

    # load data
    predict_data = get_data(predict_file, dir_prefix, data_source)
    
    n_batches_per_epoch = data_number_of_batches_per_epoch(predict_data[1], BATCH_SIZE)
    
    # get some data
    predictions = model.predict_generator(
        data_generator_train(
            predict_data[0], predict_data[2], predict_data[1],
            BATCH_SIZE, image_height, image_width, num_channels, NUM_CLASSES,
            data_source, camera_model,
            randomize=False, augment=False
        ),
        n_batches_per_epoch,
        verbose=0
    )
    

    #print predict_data[0]
    # reload data as one big batch
    all_data_loader = data_generator_train(predict_data[0], predict_data[2], predict_data[1],
            len(predict_data[1]), image_height, image_width, num_channels, NUM_CLASSES,
            data_source, camera_model,
            randomize=False, augment=False)
    all_images, all_labels = all_data_loader.next()
    

    bounding_boxes = np.zeros((all_images.shape[0],4))
    # only centroids[0] and centroids[1] will be valid. centroids[2] is added for 
    # output compatibility between lidar and camera images
    centroids = np.zeros((all_images.shape[0],3))
    timestamps = []
    ind = 0


    # extract the 'car' category labels for all pixels in the first results, 0 is non-car, 1 is car
    xyz_pred = np.zeros((all_images.shape[0], 7))
    for prediction, file_prefix in zip(predictions, predict_data[1]):
        classes = prediction[:, 1]

        classes = np.around(classes)

        #print(np.where([classes == 1.0])[1])
        #print(classes, np.max(classes))

        #obj_pixels = np.dstack((classes, classes, classes))
        #obj_pixels = np.reshape(obj_pixels, input_shape)
        obj_pixels = np.reshape(classes, (input_shape[0],input_shape[1]))

        # generate output - white pixels for car pixels
        image = obj_pixels.astype(np.uint8) * 255
                             
        centroid, bbox, bbox_area = find_obstacle(prediction, input_shape)       

        timestamp = os.path.basename(file_prefix).split('_')[0]
        
        if data_source == "lidar" and (centroid is not None) and not (centroid[0] == 0. and centroid[1] == 0.):                                                              
            print('timestamp={}'.format(timestamp))     
            centroid_3d_array = back_project_2D_2_3D(np.array([centroid]), \
                                                np.array([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]).reshape((1,4)), \
                                                np.expand_dims(all_images[ind,:,:,0], 0), \
                                                np.expand_dims(all_images[ind,:,:,1], 0))
            centroid_3d = centroid_3d_array[0]
            
            print('centroid_2d={} projected centroid_3d={}'.format(centroid, centroid_3d))
            if not (centroid_3d[0] == 0. and centroid_3d[1] == 0.):                
                pred_result, bbox_3d = find_bbox_3d(all_images[ind,:,:,0], all_images[ind,:,:,1], prediction, bbox, centroid_3d)                
                xyz_pred[ind, :] = pred_result
                
                if not np.array_equal(pred_result[:3], [0, 0, 0]):
                    # project to 2d for visualization purpose only
                    centroid_3d = pred_result[:3]
                    centroid, bbox = project_bbox_2d(pred_result, bbox_3d)                        
                    print('new centroid_2d={} centroid_3d={}'.format(centroid, centroid_3d))
                else:
                    centroid = None
            else:
                centroid = None
                
        timestamps.append(timestamp)

        if centroid is not None:       
            cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 2)            
            #print('{} -- centroid found: ({}, {}), area={}'.format(file_prefix, centroid[0], centroid[1], bbox_area))  
            centroids[ind][0] = centroid[0]
            centroids[ind][1] = centroid[1]
            centroids[ind][2] = 0
            bounding_boxes[ind,0] = bbox[0][0]
            bounding_boxes[ind,1] = bbox[0][1]
            bounding_boxes[ind,2] = bbox[1][0]
            bounding_boxes[ind,3] = bbox[1][1]               
        else:
            print('{} -- centroid not found'.format(file_prefix))
            centroids[ind][0] = 0
            centroids[ind][1] = 0
            centroids[ind][2] = 0
            bounding_boxes[ind,0] = 0
            bounding_boxes[ind,1] = 0
            bounding_boxes[ind,2] = 0
            bounding_boxes[ind,3] = 0               

        ind += 1

        if export:
            if data_source == "lidar":
                if output_dir is not None:
                    file_prefix = output_dir + "/lidar_predictions/" + os.path.basename(file_prefix)
                else:
                    file_prefix = os.path.dirname(file_prefix).replace('/lidar_360/', '') + "/lidar_predictions/" + os.path.basename(file_prefix)
            elif data_source == "camera":
                if output_dir is not None:
                    file_prefix = output_dir + "/camera_predictions/" + os.path.basename(file_prefix)
                else:
                    file_prefix = os.path.dirname(file_prefix).replace('/camera/', '') + "/camera_predictions/" + os.path.basename(file_prefix)
            
            if not(os.path.isdir(os.path.dirname(file_prefix))):
                os.mkdir(os.path.dirname(file_prefix))

            cv2.imwrite(file_prefix + "_class.png", image)
    
    if data_source == "lidar":         
        #xyz_pred = back_project_2D_2_3D(centroids, bounding_boxes, all_images[:,:,:,0], all_images[:,:,:,1])        
        return xyz_pred, timestamps
    elif data_source == "camera":
        return centroids, timestamps
    else:
        print "invalid data source type"
        exit(1)        
    
        
def main():
    parser = argparse.ArgumentParser(description='Lidar car/pedestrian trainer')
    parser.add_argument('modelFile', type=str, default="", help='Model Filename')
    parser.add_argument("predict_file", type=str, default="", help="list of data folders for prediction or rosbag file name")
    parser.add_argument('--weightsFile', type=str, default="", help='Model Weights Filename')
    parser.add_argument('--data_type', type=str, default="frontview", help='Data source for prediction: frontview, or rosbag')    
    parser.add_argument('--export', dest='export', action='store_true', help='Export images')
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--output_dir', type=str, default=None, help='output file for prediction results')
    parser.add_argument('--data_source', type=str, default="lidar", help='lidar or camera data')
    parser.add_argument('--camera_model', type=str, help='Camera calibration yaml')
    parser.add_argument('--lidar2cam_model', type=str, help='Lidar to Camera calibration yaml')
    
    parser.set_defaults(export=False)

    args = parser.parse_args()
    output_dir = args.output_dir
    data_type = args.data_type
    predict_file = args.predict_file
    dir_prefix = args.dir_prefix
    data_source = args.data_source    
    use_regression = True if data_source == "lidar" else False    
    print('data_source={} use_regression={}'.format(data_source, use_regression))

    camera_model_file = args.camera_model
    lidar2cam_model_file = args.lidar2cam_model

    camera_model = None
    prediction_file_name = None
    if data_source == "camera":
       if camera_model_file == "":
            print "need to enter camera calibration yaml"
            exit(1)
       if lidar2cam_model_file == "":
            print "need to enter lidar to camera calibration yaml"
            exit(1)
       camera_model = CameraModel()
       camera_model.load_camera_calibration(camera_model_file, lidar2cam_model_file)
       prediction_file_name = "objects_obs1_camera_predictions.csv"
    elif data_source == "lidar":
       prediction_file_name = PREDICTION_FILE_NAME
    else:
       print "invalid data source type"
       exit(1)        
    
    if data_type is None or not data_type == 'rosbag':
        data_type = 'frontview'
    print('predicting from ' + data_type)    
    
    # load model with weights
    model = load_model(args.modelFile, args.weightsFile, use_regression)
        
    if data_type == 'frontview':
        xyz_pred, timestamps = predict(model, predict_file, dir_prefix, args.export, output_dir, data_source, camera_model)        
    else:
        xyz_pred, timestamps = predict_rosbag(model, predict_file, use_regression)

    if output_dir is not None:
        file_prefix = output_dir + "/"

        write_prediction_data_to_csv(xyz_pred, timestamps, file_prefix + prediction_file_name)
        print('prediction result written to ' + file_prefix + prediction_file_name)    
        
        if use_regression:
            write_metadata(xyz_pred, file_prefix + PREDICTION_MD_FILE_NAME)
            print('metadata prediction result written to ' + file_prefix + PREDICTION_MD_FILE_NAME) 
        
if __name__ == '__main__':
    main()
