import sys
import tracket_parser
import scipy.interpolate
import csv


class TrackletInterpolater:

    def __init__(self):
        pass

    def load_timestamps_from_csv(self, timestamp_file):

        timestamps = []

        with open(timestamp_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', restval='')

            for r in reader:
                timestamp = int(r['timestamp'])
                timestamps.append(timestamp)

        return timestamps

    def interpolate_from_tracklet(self, tracklet_file, source_timestamps, dest_timestamps):

        if type(source_timestamps) is str:
            source_timestamps = self.load_timestamps_from_csv(source_timestamps)
        else:
            assert(type(dest_timestamps) is list)

        if type(dest_timestamps) is str:
            dest_timestamps = self.load_timestamps_from_csv(dest_timestamps)
        else:
            assert (type(dest_timestamps) is list)

        f = open(tracklet_file)
        data = f.read().replace('\n', '')
        f.close()

        dataDict = tracket_parser.xml_to_dict(data)
        tracklet = tracket_parser.clean_items_list(dataDict)

        tracket_parser.put_timestamps_with_frame_ids(tracklet, source_timestamps)

        return self.interpolate(tracklet, dest_timestamps), tracklet

    def interpolate_from_csv(self, csv_file, timestamps):

        points = []

        if type(timestamps) is str:
            timestamps = self.load_timestamps_from_csv(timestamps)
        else:
            assert(type(timestamps) is list)

        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', restval='')

            for r in reader:
                timestamp = int(r['timestamp'])
                tx = float(r['tx'])
                ty = float(r['ty'])
                tz = float(r['tz'])

                points.append({'timestamp': timestamp, 'tx': tx, 'ty': ty, 'tz': tz})

        return self.interpolate(points, timestamps)

    def interpolate(self, source, dest_timestamps):

        timestamps = list(map(lambda x: x['timestamp'], source))
        txs = list(map(lambda x: x['tx'], source))
        tys = list(map(lambda x: x['ty'], source))
        tzs = list(map(lambda x: x['tz'], source))        
        rzs = list(map(lambda x: x['rz'], source))
        
        fx = scipy.interpolate.interp1d(timestamps, txs, fill_value='extrapolate')
        fy = scipy.interpolate.interp1d(timestamps, tys, fill_value='extrapolate')
        fz = scipy.interpolate.interp1d(timestamps, tzs, fill_value='extrapolate')       
        frz = scipy.interpolate.interp1d(timestamps, rzs, fill_value='extrapolate')

        dest_data = []

        for dest_timestamp in dest_timestamps:
            dest_data.append({'timestamp': dest_timestamp,
                              'tx': fx(dest_timestamp),
                              'ty': fy(dest_timestamp),
                              'tz': fz(dest_timestamp),
                              'rx': 0, 'ry': 0, 
                              'rz': frz(dest_timestamp)})

        return dest_data
