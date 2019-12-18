import sys
import os
import json
import rosbag


from xml.etree.ElementTree import fromstring
from xmljson import Parker

xp = Parker()

def usage():
    print('Converts a tracklets XML file to JSON')
    print('Usage: python parser.py [tracklet labels XML] [rosbag file]')

def xml_to_dict(data):
    return xp.data(fromstring(data))

def clean_items_list(data):
    tracklets = data.get('tracklets', {})
    if type(tracklets.get('item')) is not list:
        item = tracklets.get('item', {})
        cleaned = []
        objID = 0
        objType = item.get('objectType', '')
        first_frame = item.get('first_frame', 0)
        h, w, l = item.get('h', 0), item.get('w', 0), item.get('l', 0)
        for frame, pose in enumerate(item.get('poses', {}).get('item', [])):
            cleaned.append({
                'object_id': objID,
                'object_type': objType,
                'frame_id': first_frame + frame,
                'tx': pose['tx'],
                'ty': pose['ty'],
                'tz': pose['tz'],
                'rx': pose['rx'],
                'ry': pose['ry'],
                'rz': pose['rz'],
                'width': w,
                'height': h,
                'depth': l,
            })
    else:
        items = tracklets.get('item', [])
        cleaned = []
        for count, item in enumerate(items):
            objID = count
            objType = item.get('objectType', '')
            first_frame = item.get('first_frame', 0)
            h, w, l = item.get('h', 0), item.get('w', 0), item.get('l', 0)
            for frame, pose in enumerate(item.get('poses', {}).get('item', [])):
                cleaned.append({
                    'object_id': objID,
                    'object_type': objType,
                    'frame_id': first_frame + frame,
                    'tx': pose['tx'],
                    'ty': pose['ty'],
                    'tz': pose['tz'],
                    'rx': pose['rx'],
                    'ry': pose['ry'],
                    'rz': pose['rz'],
                    'width': w,
                    'height': h,
                    'depth': l,
                })
    return cleaned

#
# get timestamps of each image frame. currently using topic name
# '/image_raw' to count the frames.
# Kitti vs Udacity data will have a different topic due to one color camera
#
def get_camera_frame_timestamps(bag):

   timestamps = []
   topicTypesMap = bag.get_type_and_topic_info().topics

   for topic, msg, t in bag.read_messages(topics=['/image_raw']):

       msgType = topicTypesMap[topic].msg_type
       assert(msgType == 'sensor_msgs/Image')
       timestamps.append(t.to_sec())

   return timestamps

#
# insert timestamp field into dictonary items by using frame ids
#
def put_timestamps_with_frame_ids(data, timestamps):
    for items in data:
        frame_id = items['frame_id']
        item_timestamp = timestamps[frame_id]
        items['timestamp'] = item_timestamp

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        sys.exit()

    try:
        f = open(sys.argv[1])
        data = f.read().replace('\n', '')
        f.close()

    except:
        print('Unable to read file: %s' % sys.argv[1])
        f.close()
        sys.exit()

    try:
        bag = rosbag.Bag(sys.argv[2], 'r')
    except:
        print('Unable to read file: %s' % sys.argv[2])
        f.close()
        sys.exit()



    dataDict = xml_to_dict(data)

    cleaned = clean_items_list(dataDict)

    timestamps = get_camera_frame_timestamps(bag)

    put_timestamps_with_frame_ids(cleaned, timestamps)

    dataJson = json.dumps({
        'data': cleaned,
    })


    print(dataJson)
