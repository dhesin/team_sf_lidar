import argparse
import numpy as np
import rosbag
import sensor_msgs.point_cloud2
import matplotlib as mpl
mpl.use('Agg')  #Skip using X11
import matplotlib.pyplot as plt
import matplotlib
from itertools import repeat


startsec = 0
LIDAR_MAX_DENSITY_SQ_METER=1500

def generate_value_channel(x_range, y_range, width_grid_length, height_grid_length):
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    dim = (int(width/width_grid_length), int(height/height_grid_length))
    return np.zeros(dim)


def produce_view(channel, name, t):
    fig = plt.figure()
    plt.imshow(channel)
    fig.savefig(name + '_' + str(t) + '_plot.png')


# The point cloud density
# indicates the number of points in each cell. To normalize
# the feature, it is computed as min(1.0, log(N+1)/log(64) ), where N is the number of points in the cell.
def normalize(channel):
    log64 = np.log(64)
    for i, r in enumerate(channel):
        for j, c in enumerate(r):
            if channel[i][j] > 0:
                channel[i][j] = 255 * min(np.log(channel[i][j] + 1) / log64, 1)
    return channel


# generate birds view for one frame
def generate_birds_eye_view(arrPoints, t, res, max_range=120, cmap='gray'):
    points = np.array(arrPoints)
    bins = (np.arange(-max_range, max_range, res[1]), np.arange(-max_range, max_range, res[0]))
    density_channel, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=bins)
    density_norm = normalize(density_channel)
    density_norm = np.flipud(np.fliplr(density_norm))

    colormap = matplotlib.cm.ScalarMappable(cmap=cmap,
                                            norm=matplotlib.colors.Normalize(
                                                vmin=0,
                                                vmax=LIDAR_MAX_DENSITY_SQ_METER*res[0])
                                            )
    retval = colormap.to_rgba(density_norm, bytes=True, norm=True)[:,:,0:3]
    return retval


def load(topic, msg, time):
    t = time.to_sec()
    since_start = msg.header.stamp.to_sec()-startsec
    arrPoints = []
    if topic in ['/radar/points','/velodyne_points']:
        points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)
        for point in points:
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]
            arrPoints.append(point[:4])
    return arrPoints


def read(dataset, skip, topics_list):
    """
    return an image of
    """
    startsec = 0

    print("reading rosbag ", dataset)
    bag = rosbag.Bag(dataset, 'r')
    for topic, msg, t in bag.read_messages(topics=topics_list):
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
                points = load(topic, msg, t)
        img = generate_birds_eye_view(points, t)
        produce_view(img, "density_channel", t)

# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="appTitle")
    parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROS Bag name')
    parser.add_argument('--skip', type=int, default="0", help='skip seconds')
    args = parser.parse_args()
    dataset = args.dataset
    skip = args.skip

    topics_list = [
      '/velodyne_points'
    ]
    read(dataset, skip, topics_list)
