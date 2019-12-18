import sys
sys.path.append('../')
import csv
import glob
import argparse
import os.path
import math
from common.interpolate import TrackletInterpolater
from common.tracklet_generator import Tracklet, TrackletCollection

def main():

    appTitle = "Udacity Team-SF: Tracklet generator"
    parser = argparse.ArgumentParser(description=appTitle)
    parser.add_argument('pred_csv', type=str, help='Prediction CSV')
    parser.add_argument('camera_csv', type=str, help='Camera timestamps CSV')
    parser.add_argument('metadata', type=str, help='Metadata File')
    parser.add_argument('out_xml', type=str, help='Tracklet File')
    parser.add_argument('--offset_csv', type=str, default=None, help='Offset factor')

    args = parser.parse_args()

    offset = None

    csvfile = open(args.metadata, 'r')
    reader = csv.DictReader(csvfile, delimiter=',')
    tracklet = None
    tracklet_xml = TrackletCollection()
    for mdr in reader:
        tracklet = Tracklet('Car', float(mdr['l']), float(mdr['w']), float(mdr['h']))

    if args.offset_csv is not None:
        csvfile = open(args.offset_csv, 'r')
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            offset = row

    interpolater = TrackletInterpolater()
    interpolated_camera = interpolater.interpolate_from_csv(args.pred_csv, args.camera_csv)

    for i in range(len(interpolated_camera)):
        if offset is not None:
            interpolated_camera[i]['tx'] += float(offset['tx'])
            interpolated_camera[i]['ty'] += float(offset['ty'])
            interpolated_camera[i]['tz'] += float(offset['tz'])
            
        if(math.isnan(interpolated_camera[i]['tx'])):
            if(i > 0):
                interpolated_camera[i]['tx'] = interpolated_camera[i - 1]['tx'] 
                interpolated_camera[i]['ty'] = interpolated_camera[i - 1]['ty'] 
                interpolated_camera[i]['tz'] = interpolated_camera[i - 1]['tz']
            else:
                j = i + 1
                while (len(interpolated_camera) > j and math.isnan(interpolated_camera[j]['tx'])):
                    j = j + 1
                if(len(interpolated_camera) > j):
                    interpolated_camera[i]['tx'] = interpolated_camera[j]['tx'] 
                    interpolated_camera[i]['ty'] = interpolated_camera[j]['ty'] 
                    interpolated_camera[i]['tz'] = interpolated_camera[j]['tz'] 
                else:
                    interpolated_camera[i]['tx'] = 0 
                    interpolated_camera[i]['ty'] = 0 
                    interpolated_camera[i]['tz'] = 0

    tracklet.poses = interpolated_camera
    tracklet_xml.tracklets = [tracklet]
    tracklet_xml.write_xml(args.out_xml)


if __name__ == '__main__':
    main()
