import sys
sys.path.append('../')
import pyglet
import argparse
import numpy as np
import rosbag
import os
import matplotlib.image as mpimg
import sensor_msgs.point_cloud2
import csv
import cv2
import globals
import common.camera_model
from cv_bridge import CvBridge

from extract_rosbag_lidar import generate_lidar_2d_front_view
from extract_rosbag_lidar import save_lidar_2d_images
from rectify_image import extract_calib_info
from rectify_image import initUndistortRectifyMap
from rectify_image import remap
from common.birds_eye_view_generator import generate_birds_eye_view
from common.interpolate import TrackletInterpolater

import radar_tracks

class ROSBagExtractor:

    def __init__(self,
                 window_max_width=875,
                 topdown_res=.2,
                 topdown_max_range=120,
                 cmap=None,
                 output_dir=None,
                 display=False,
                 pickle=False,
                 yaml_path=None):
        self.windows = {}
        self.bridge = CvBridge()
        self.window_max_width = window_max_width
        self.cmap = cmap
        self.output_dir = output_dir
        self.topdown_res = (topdown_res, topdown_res)
        self.topdown_max_range = topdown_max_range
        self.display=display
        self.pickle=pickle
        self.lidar_timestamps=[]
        self.camera_timestamps = []
        self.radar_tracks = []
        self.yaml_path = yaml_path
        self.camera_model = None

        if output_dir is not None:
            if not(os.path.isdir(self.output_dir + '/lidar_360/')):
                os.makedirs(self.output_dir + '/lidar_360/')
            if not (os.path.isdir(self.output_dir + '/topdown/')):
                os.makedirs(self.output_dir + '/topdown/')
            if not (os.path.isdir(self.output_dir + '/camera/')):
                os.makedirs(self.output_dir + '/camera/')
            if not (os.path.isdir(self.output_dir + '/radar/')):
                os.makedirs(self.output_dir + '/radar/')

        if self.yaml_path is None:
            print('yaml_path is not provided. So the output images will not be rectified')
        elif not os.path.isfile(self.yaml_path):
            print('yaml_path ' + self.yaml_path + ' does not exist. So the output images will not be rectified')
        else:
            self.camera_model = common.camera_model.CameraModel()
            self.camera_model.load_camera_calibration(self.yaml_path)
        

    @staticmethod
    def save_image(output_dir, name, count, image, camera_model=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if camera_model is not None:
            image = camera_model.rectify_image(image)
        cv2.imwrite('{}/{}_{}.png'.format(output_dir, count, name), image[globals.CAM_IMG_TOP:globals.CAM_IMG_BOTTOM,:,:])

    @staticmethod
    def print_msg(msgType, topic, msg, time, startsec):
        t = time.to_sec()
        since_start = 0

        if 'sensor_msgs' in msgType or 'nav_msgs' in msgType:
            since_start = msg.header.stamp.to_sec() - startsec

        if msgType == 'sensor_msgs/PointCloud2':
            print(topic, msg.header.seq, since_start, 'nPoints=', msg.width * msg.height, t)

        elif msgType == 'sensor_msgs/NavSatFix':
            print(topic, msg.header.seq, since_start, msg.latitude, msg.longitude, msg.altitude, t)

        elif msgType == 'nav_msgs/Odometry':

            position = msg.pose.pose.position
            print(topic, msg.header.seq, since_start, position.x, position.y, position.z, t)

        elif msgType == 'sensor_msgs/Range':

            print(topic, msg.header.seq, since_start, msg.radiation_type, msg.field_of_view, msg.min_range, msg.max_range,
                  msg.range, t)

        elif msgType == 'sensor_msgs/Image':

            print(topic, msg.header.seq, msg.width, msg.height, since_start, t)

        elif msgType == 'sensor_msgs/CameraInfo':

            print(topic, msg.header.seq, since_start, msg.width, msg.height, msg.distortion_model, t)

        else:
            pass
            # print(topic, msg.header.seq, t-msg.header.stamp, msg, t)

    def get_window(self, topic, img):
        if self.windows.get(topic, None) is None:
            print(img.shape)
            ratio = self.window_max_width / float(img.shape[1])
            size = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))
            self.windows[topic] = pyglet.window.Window(size[0], size[1], caption=topic)
        return self.windows[topic]

    @staticmethod
    def save_images(output_dir, count, images):
        for k, img in images.iteritems():
            mpimg.imsave('{}/{}_{}.png'.format(output_dir, count, k), images[k], origin='upper')

    @staticmethod
    def convert_img(img):
        return pyglet.image.ImageData(img.shape[1], img.shape[0], 'RGB', np.flipud(img).tobytes())

    def handle_msg(self, msg_type, topic, msg, timestamp, result):

        window = []
        img = []

        if '/radar/tracks' in topic:
            tracks = radar_tracks.parse_msg(msg, timestamp)
            if len(tracks) > 0:
                self.radar_tracks += tracks

        elif msg_type in ['sensor_msgs/Image']:

            self.camera_timestamps.append(timestamp.to_nsec())

            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            if self.display:
                window.append(self.get_window(topic, cv_img))
                img.append(self.convert_img(cv_img))

            name = 'image'
            if 'center' in topic:
                name = 'center'
            elif 'left' in topic:
                name = 'left'
            elif 'right' in topic:
                name = 'right'

            if self.output_dir is not None:
                self.save_image(self.output_dir + '/camera/', name, timestamp, cv_img, self.camera_model)

        elif msg_type in ['sensor_msgs/PointCloud2'] and 'velo' in topic:           
            self.lidar_timestamps.append(timestamp.to_nsec())

            points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=False)
            points = np.array(list(points))

            # render top down point cloud
            density_top_down = generate_birds_eye_view(points, timestamp, res=self.topdown_res, max_range=self.topdown_max_range, cmap=self.cmap)
            density_top_down = density_top_down.astype(np.uint8)

            # render 360 view
            lidar_images = generate_lidar_2d_front_view(points, cmap=self.cmap)

            if not(self.pickle):
                del lidar_images['intensity_float']
                del lidar_images['distance_float']
                del lidar_images['height_float']

            result['intensity'][str(timestamp)] = lidar_images['intensity']
            result['distance'][str(timestamp)] = lidar_images['distance']
            result['height'][str(timestamp)] = lidar_images['height']

            # save files
            if self.output_dir is not None:
                save_lidar_2d_images(self.output_dir + '/lidar_360/', timestamp.to_nsec(), lidar_images)
                self.save_images(self.output_dir + '/topdown/', timestamp.to_nsec(), {'density': density_top_down})

            if self.display:

                img.extend(
                    map(self.convert_img, [
                        lidar_images['intensity'],
                        lidar_images['distance'],
                        lidar_images['height'],
                        density_top_down
                    ])
                )

                window.extend([
                    self.get_window(topic + '/360/intensity', lidar_images['intensity']),
                    self.get_window(topic + '/360/distance', lidar_images['distance']),
                    self.get_window(topic + '/360/height', lidar_images['height']),
                    self.get_window(topic + '/topdown/density', density_top_down),
                ])

        if self.display:
            for w, i in zip(window, img):
                w.switch_to()
                w.dispatch_events()
                size = w.get_size()
                i.blit(0, 0, width=size[0], height=size[1])
                w.flip()

    def save_radar_tracks(self):
        if len(self.radar_tracks) == 0:
            return

        with open(os.path.join(self.output_dir, 'radar', 'radar_tracks.csv'), 'wb') as output_file:
            writer = csv.DictWriter(output_file, self.radar_tracks[0].keys())
            writer.writeheader()
            writer.writerows(self.radar_tracks)

def write_timestamps_to_csv(timestamps, output_file):       
    csv_file = open(output_file, 'w')
    writer = csv.DictWriter(csv_file, ['timestamp'])

    writer.writeheader()
    
    for ts in timestamps:
        writer.writerow({'timestamp': ts})


def main():

    appTitle = "Udacity Team-SF: ROSbag viewer"
    parser = argparse.ArgumentParser(description=appTitle)
    parser.add_argument('bag_file', type=str, help='ROS Bag name')
    parser.add_argument('--skip', type=float, default="0", help='skip seconds')
    parser.add_argument('--length', type=float, default=None, help='length seconds')
    parser.add_argument('--display', dest='display', action='store_true', help='Display output')
    parser.add_argument('--topics', type=str, default=None, help='Topic list to display')
    parser.add_argument('--topdown_res', type=float, default=2, help='Topdown image fidelity (meters/pixel)')
    parser.add_argument('--topdown_max_range', type=float, default=120, help='Topdown max range (meters)')
    parser.add_argument('--lidar_cmap', type=str, default='jet', help='Colormap for lidar images (Default "jet")')
    parser.add_argument('--pickle', dest='pickle', default=None, action='store_true', help='Export pickle files')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for images')
    parser.add_argument('--quiet', dest='quiet', action='store_true', help='Quiet mode')
    parser.add_argument('--interpolate', type=str, dest='interpolate', help='Interpolate with tracklet file')
    parser.add_argument('--camera_calibration', type=str, default=None,  help='Yaml for camera image rectification')
    parser.set_defaults(quiet=False, display=False)

    args = parser.parse_args()

    bag_file = args.bag_file
    output_dir = args.outdir

    if not os.path.isfile(bag_file):
        print('bag_file ' + bag_file + ' does not exist')
        sys.exit()

    if output_dir is not None and not(os.path.isdir(output_dir)):
        print('output_dir ' + output_dir + ' does not exist')
        sys.exit()

    skip = args.skip
    length = args.length
    startsec = 0
    last_topic_time = {}
    maximum_gap_topic = {}
    topics_list = args.topics.split(',') if args.topics else None

    extractor = ROSBagExtractor(cmap=args.lidar_cmap,
                                output_dir=output_dir,
                                topdown_res=args.topdown_res,
                                topdown_max_range=args.topdown_max_range,
                                pickle=args.pickle,
                                display=args.display,
                                yaml_path=args.camera_calibration)

    print("reading rosbag ", bag_file)
    bag = rosbag.Bag(bag_file, 'r')
    topicTypesMap = bag.get_type_and_topic_info().topics

    result = {'intensity': {}, 'distance': {}, 'height': {}}
    for topic, msg, t in bag.read_messages(topics=topics_list):
        msgType = topicTypesMap[topic].msg_type
        if startsec == 0:
            startsec = t.to_sec()
            if skip < 24 * 60 * 60:
                skipping = t.to_sec() + skip
                print("skipping ", skip, " seconds from ", startsec, " to ", skipping, " ...")
            else:
                skipping = skip
                print("skipping to ", skip, " from ", startsec, " ...")
        else:
            if t.to_sec() > skipping:

                if length is not None and t.to_sec() > skipping + length:
                    break

                if last_topic_time.get(topic) != None:
                    gap = t.to_sec() - last_topic_time[topic]
                    if maximum_gap_topic.get(topic) == None or gap > maximum_gap_topic[topic]:
                        maximum_gap_topic[topic] = gap

                last_topic_time[topic] = t.to_sec()

                if not args.quiet:
                    extractor.print_msg(msgType, topic, msg, t, startsec)
                if args.display or output_dir:
                    extractor.handle_msg(msgType, topic, msg, t, result)

    extractor.save_radar_tracks()
    
    # remove duplicates from lidar timestamps -- there is probably a bug in bag processing    
    extractor.lidar_timestamps = sorted(set(extractor.lidar_timestamps))    
    
    # export timestamps
    write_timestamps_to_csv(extractor.lidar_timestamps, output_dir + '/lidar_timestamps.csv')
    write_timestamps_to_csv(extractor.camera_timestamps, output_dir + '/camera_timestamps.csv')   
    
    # generate lidar interpolation
    if args.interpolate:

        print('interpolating...')

        interpolater = TrackletInterpolater()
        lidar_ground_truth, camera_ground_truth = interpolater.interpolate_from_tracklet(
            args.interpolate,
            extractor.camera_timestamps,
            extractor.lidar_timestamps
        )

        with open(output_dir + '/obs_poses_interp_transform.csv', 'w') as csvfile:

            writer = csv.DictWriter(csvfile, ['timestamp', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'], delimiter=',', restval='', )

            writer.writeheader()
            writer.writerows(lidar_ground_truth)

        with open(output_dir + '/obs_poses_camera.csv', 'w') as csvfile:

            writer = csv.DictWriter(csvfile, ['timestamp', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'], extrasaction='ignore', delimiter=',', restval='', )

            writer.writeheader()
            writer.writerows(camera_ground_truth)

    #Load pickle:
    #input
    '''
    f = open(output_dir + '/lidar.p', 'rb')
    pickle_data = pickle.load(f)
    print(pickle_data)
    for mapType, pointMap in pickle_data.items():
        print(mapType)
        for t, value in pointMap.items():
            print(t)
            print(np.shape(value))
    '''

    #output
    '''
    distance
    1490149174663355139
    (93, 1029)
    ...
    intensity
    1490149174663355139
    (93, 1029)
    ...
    height
    1490149174663355139
    (93, 1029)
    ...

    '''
    print("Max interval between messages per topic")
    for key, value in sorted(maximum_gap_topic.iteritems(), key=lambda (k,v): (v,k)):
        print("    {}: {}".format(key, value))

# ***** main loop *****
if __name__ == "__main__":
    main()
