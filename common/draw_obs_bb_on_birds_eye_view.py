#!/usr/bin/python
"""
draw_obs_bb_on_birds_eye_view.py: version 0.1.0
Note:

Todo:

History:
"""

import argparse
import numpy as np
import rosbag

appTitle = "Udacity Team-SF: Obstacle bounding boxes drawer on bird's eye views"

def print_obs_bb(msgType, topic, msg, time, capture_vehicle_position, obstacle_list, obstacle_positions):

    t = time.to_sec()

    if 'sensor_msgs' in msgType or 'nav_msgs' in msgType:
        since_start = msg.header.stamp.to_sec()-startsec

    if msgType in ['nav_msgs/Odometry']:
        print(topic, msg.header.seq, since_start, msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, t)
        for i  in range(len(obstacle_list)):
            print('Relative position of ', obstacle_list[i], ' with respect to the capture vehicle:', obstacle_positions[i][0]-capture_vehicle_position[0], obstacle_positions[i][1]-capture_vehicle_position[1], obstacle_positions[i][2]-capture_vehicle_position[2], t)

# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=appTitle)
  parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROS Bag name')
  parser.add_argument('--skip', type=int, default="0", help='skip seconds')
  parser.add_argument('--topics', type=str, default=None, help='Topic list to display')
  #The argument for the resolution for the bird's eye view image is in cm per pixel
  parser.add_argument('--resolution', type=float, default=10.0, help='Resolution per pixel')  
  args = parser.parse_args()

  dataset = args.dataset
  skip = args.skip
  resolution = args.resolution
  
  startsec = 0
  topics_list = args.topics.split(',') if args.topics else None

  print("reading rosbag ", dataset)
  bag = rosbag.Bag(dataset, 'r')
  topicTypesMap = bag.get_type_and_topic_info().topics

  #SN: Add obstacle list
  obstacle_list = []
  seen_vehicle_firsttime = False
  capture_vehicle_position = [0, 0, 0]
  obstacle_positions = []

  for topic, msg, t in bag.read_messages(topics=topics_list):
    msgType = topicTypesMap[topic].msg_type
    if startsec == 0:
        startsec = t.to_sec()
        if skip < 24*60*60:
            skipping = t.to_sec() + skip
            print("skipping ", skip, " seconds from ", startsec, " to ", skipping, " ...")
        else:
            skipping = skip
            print("skipping to ", skip, " from ", startsec, " ...")
    else:
        if t.to_sec() > skipping:
            #SN:
            if not(seen_vehicle_firsttime):
                if (msgType in ['nav_msgs/Odometry']) & (topic == '/gps/rtkfix'):
                    seen_vehicle_firsttime = True
                    capture_vehicle_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            else:
                if (msgType in ['nav_msgs/Odometry']) & (topic == '/gps/rtkfix'):
                    capture_vehicle_position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                    print_obs_bb(msgType, topic, msg, t, capture_vehicle_position, obstacle_list, obstacle_positions)
                elif (msgType in ['nav_msgs/Odometry']):
                    if topic in obstacle_list:
                        for i in range(len(obstacle_list)):
                            if topic == obstacle_list:
                                break
                        obstacle_positions[i] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                        print_obs_bb(msgType, topic, msg, t, capture_vehicle_position, obstacle_list, obstacle_positions)
                    else:
                        obstacle_list.append(topic)
                        obstacle_positions.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                        print_obs_bb(msgType, topic, msg, t, capture_vehicle_position, obstacle_list, obstacle_positions)                        
