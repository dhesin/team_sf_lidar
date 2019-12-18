import pandas as pd
import sys
import math
import os
sys.path.append('../')


from common.camera_model import CameraModel, generateImage


def get_radar_tracks(radar_track_file):
    radar_data = pd.read_csv(radar_track_file,header=0)
    radar_data_time_based = radar_data.groupby('timestamp')
    return radar_data_time_based


def get_nearest_camera_timestamp(radar_ts, camera_timestamps):
    return min(camera_timestamps, key=lambda x: abs(int(x) - int(radar_ts)))

def write_radar_data_to_csv(data, output_file):
    import csv
    csv_file = open(output_file, 'w')
    csv_file.write('radar_ts,'+'instance,'+'img_ts,'+'distance,'+'angle,'+'tx,'+ 'ty,'+ 'tz\n')
    for item in data:
        csv_file.write(item+'\n')
    csv_file.close()
    print 'Radar processed output saved at: {}'.format(output_file)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Radar data processor')
    parser.add_argument('--radar_tracks',type=str, help='radar_tracks csv')
    parser.add_argument('--camera', type=str, help='Camera calibration yaml')
    parser.add_argument('--lidar', type=str, help='Lidar to Camera calibration yaml')
    parser.add_argument('--timestamps', type=str, help='Camera timestamps csv')
    parser.add_argument('--input_dir', type=str, help='Rectified camera images directory')
    parser.add_argument('--output_dir', type=str, help='Annotation camera images directory')
    parser.add_argument('--radar_out_file', type=str, help='Processed radar output csv')

    args = parser.parse_args()
    radar_tracks =  args.radar_tracks
    camera_calibration = args.camera
    lidar_camera_calibration = args.lidar
    camera_timestamps = args.timestamps
    input_dir = args.input_dir
    out_dir = args.output_dir
    radar_out_file = args.radar_out_file

    if not os.path.isfile(radar_tracks):
        print('radar_tracks does not exist')
        sys.exit()
    if not os.path.isfile(camera_calibration):
        print('camera_calibration does not exist')
        sys.exit()
    if not os.path.isfile(lidar_camera_calibration):
        print('lidar_camera_calibration does not exist')
        sys.exit()
    if not os.path.isfile(camera_timestamps):
        print('camera_timestamps does not exist')
        sys.exit()
    if not os.path.isdir(input_dir):
        print('input_dir does not exist')
        sys.exit()
    if not os.path.isdir(out_dir):
        print('out_dir does not exist')
        sys.exit()
    if radar_out_file is None:
        print('radar_out_file is empty')
        sys.exit()

    #load camera model
    camera = CameraModel()
    camera.load_camera_calibration(camera_calibration, lidar_camera_calibration)

    #get readar tracks groupby timestamp
    radar_tracks_ts_based = get_radar_tracks(radar_tracks)

    #get camera time stamp
    try:
        f = open(camera_timestamps)
        import csv
        csv_reader = csv.DictReader(f, delimiter=',', restval='')
        cam_timestamps = []
        for row in csv_reader:
            cam_timestamps.append(row['timestamp'])
        f.close()

    except:
        print('Unable to read file: %s' % camera_timestamps)
        f.close()
        exit(-1)

    #approx car/obstacle dimension
    w = 1.447800
    h = 1.574800
    l = 4.241800

    radar_lidar_offset = 3.8 - 1.5494
    out_write = []

    for key,item in radar_tracks_ts_based:
        #get nearest camera timestamp
        ts = get_nearest_camera_timestamp(key,cam_timestamps)
        i = 0
        for index, radar_instance in item.iterrows():
            i = i+1

            distance = radar_instance['range']
            angle_deg = radar_instance['angle']
            theata = math.radians(angle_deg)
            distance += radar_lidar_offset

            tx = distance * math.cos(theata)
            ty = - distance * math.sin(theata)
            tz = 0 #-1.27

            str_out = 'radar_ts: {},'.format(key) + 'instance: {},'.format(i) + 'img_ts: {},'.format(
                ts) + 'distance: {},'.format(distance) + 'angle: {} deg,'.format(angle_deg) + 'tx: {},'.format(
                tx) + 'ty: {},'.format(ty) + 'tz: {},'.format(tz)

            out_write.append(str_out)
            bbox = []
            centroid = [tx, ty, tz, 0.0]

            bbox.append(centroid)
            bbox.append([tx - l / 2., ty + w / 2., tz + h / 2., 0.0])
            bbox.append([tx - l / 2., ty - w / 2., tz + h / 2., 0.0])
            bbox.append([tx + l / 2., ty + w / 2., tz + h / 2., 0.0])
            bbox.append([tx + l / 2., ty - w / 2., tz + h / 2., 0.0])
            bbox.append([tx + l / 2., ty - w / 2., tz - h / 2., 0.0])
            bbox.append([tx - l / 2., ty + w / 2., tz - h / 2., 0.0])
            bbox.append([tx - l / 2., ty - w / 2., tz - h / 2., 0.0])
            bbox.append([tx + l / 2., ty + w / 2., tz - h / 2., 0.0])
            if(i==1):
                generateImage(camera, bbox,
                              '{}/image_{}.png'.format(input_dir, ts),
                              '{}/image_{}.png'.format(out_dir, ts))#'''
            else:
                generateImage(camera, bbox,
                              '{}/image_{}.png'.format(out_dir, ts),
                              '{}/image_{}.png'.format(out_dir, ts))

    write_radar_data_to_csv(out_write,'{}/{}'.format(out_dir, radar_out_file))

if __name__ == '__main__':
    main()
