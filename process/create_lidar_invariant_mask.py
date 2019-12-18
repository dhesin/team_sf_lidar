#!/usr/bin/python
"""Create lidar invariant pixel mask from processed rosbags.
   usage: python create_lidar_invariant_mask.py input_dir output_dir
"""

import os
import sys
import argparse
import numpy as np
import rosbag
import sensor_msgs.point_cloud2
import matplotlib.image as mpimg
import pickle
import globals
import fnmatch

from extract_rosbag_lidar import generate_lidar_2d_front_view

intensity_reference = {}
invariant_intensity_array = np.ones((globals.Y_MAX + 1, globals.X_MAX + 1),dtype=bool)

height_reference = {}
invariant_height_array = np.ones((globals.Y_MAX + 1, globals.X_MAX + 1),dtype=bool)

distance_reference = {}
invariant_distance_array = np.ones((globals.Y_MAX + 1, globals.X_MAX + 1),dtype=bool)


#Initializing reference array. This method shall be called at frame 1
def initialize_invariant_pixels_new(float_pixels,type):
    global intensity_reference, height_reference, distance_reference,invariant_reference

    if type == "distance":
        intensity_reference = float_pixels
    elif type == "height":
        height_reference = float_pixels
    elif type == "intensity":
        distance_reference = float_pixels
    else:
        print "invalid type"
        return

#updating the mask. Shall be called for each frame
def update_invariant_pixels_new(float_pixels,type):
    global intensity_reference, height_reference, distance_reference
    global invariant_intensity_array, invariant_height_array, invariant_distance_array

    if type == "distance":
        invariant_reference = intensity_reference
        mask_array = invariant_intensity_array
    elif type == "height":
        invariant_reference = height_reference
        mask_array = invariant_height_array
    elif type == "intensity":
        invariant_reference = distance_reference
        mask_array = invariant_distance_array
    else:
        print "invalid type"
        return

    diff_bool_array = invariant_reference == float_pixels
    np.logical_and(mask_array, diff_bool_array,mask_array)

def main():
    """Extract velodyne points and create lidar invariant mask from a ROS bags
    """
    parser = argparse.ArgumentParser(description="Extract velodyne points and project to 2D images from a ROS bag.")
    parser.add_argument("input_dir", help="Input ROS bag directory.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--cmap", help="Color Map.", default='jet')

    frame = 0

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(input_dir):
        print('input_dir ' + input_dir + ' does not exist')
        sys.exit()

    if not os.path.isdir(output_dir):
        print('output_dir ' + output_dir + ' does not exist')
        sys.exit()

    for root, dirnames, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, '*.bag'):
            bag_file = os.path.join(root, filename)

            print("Extract velodyne_points from {}".format(bag_file))

            bag = rosbag.Bag(bag_file, "r")
            print "Finding the pixels for mask"
            for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
                points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=False)
                points = np.array(list(points))
                images = generate_lidar_2d_front_view(points, cmap=args.cmap)
                if frame == 0:

                    initialize_invariant_pixels_new(images['distance_float'], 'distance')
                    initialize_invariant_pixels_new(images['height_float'], 'height')
                    initialize_invariant_pixels_new(images['intensity_float'], 'intensity')
                else:

                    update_invariant_pixels_new(images['distance_float'], 'distance')
                    update_invariant_pixels_new(images['height_float'], 'height')
                    update_invariant_pixels_new(images['intensity_float'], 'intensity')
                frame = frame+1
                #break
            bag.close()


    f = open(output_dir + '/invariant_intensity.p', 'wb')
    pickle.dump(invariant_intensity_array, f)
    f.close()

    f = open(output_dir + '/invariant_distance.p', 'wb')
    pickle.dump(invariant_distance_array, f)
    f.close()

    f = open(output_dir + '/invariant_height.p', 'wb')
    pickle.dump(invariant_height_array, f)
    f.close()

    img = np.array(invariant_distance_array)
    mpimg.imsave('{}/distance.png'.format(output_dir), img, origin='upper')

    img = np.array(invariant_intensity_array)
    mpimg.imsave('{}/intensity.png'.format(output_dir), img, origin='upper')

    img = np.array(invariant_height_array)
    mpimg.imsave('{}/height.png'.format(output_dir), img, origin='upper')

    return

if __name__ == '__main__':
    main()
