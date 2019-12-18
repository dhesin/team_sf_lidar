# parse a /radar/tracks msg from a rosbag into a list of dicts

#usage example:
#
#import radar_tracks
#
#if '/radar/tracks' in topic:
#    tracks = radar_tracks.parse_msg(msg, timestamp)

def parse_msg(msg, timestamp):

    tracks = []

    try:
        for track in msg.tracks:
            tracks.append({
                'timestamp': timestamp, # XXX use msg header timestamp instead?
                'status': track.status,
                'number': track.number,
                'range': track.range,
                'rate': track.rate,
                'accel': track.accel,
                'angle': track.angle,
                'width': track.width,
                'late_rate': track.late_rate,
                'moving': track.moving,
                'power': track.power,
                'absolute_rate': track.absolute_rate,
            })
    except AttributeError:
        print('Not a valid /radar/tracks msg')

    return tracks

