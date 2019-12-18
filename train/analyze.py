import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colors
from common.csv_utils import foreach_dirset, load_lidar_interp

def convert_to_polar(transforms):
    arr = []
    for item in transforms:
        x, y = np.float64(item['tx']), np.float64(item['ty'])
        arr.append(
            {
                'rho': np.sqrt(x**2+y**2),
                'phi': np.arctan2(y,x) * 180 / np.pi
            }
        )
    return arr

def main():

    parser = argparse.ArgumentParser(description='Lidar car/pedestrian analyzer')
    parser.add_argument("input_csv_file", type=str, default="../data/data_folders.csv", help="list of data folders for training")
    parser.add_argument("--dir_prefix", type=str, help="absolute path to folders")

    args = parser.parse_args()
    input_csv_file = args.input_csv_file
    dir_prefix = args.dir_prefix

    x = []
    y = []

    def process(dirset):

        # load lidar ground truth
        lidar_coord = load_lidar_interp(dirset.dir)

        # determine polar coordinates
        polar_coord = convert_to_polar(lidar_coord)

        # generate histogram
        x.extend(list(map(lambda x: x['rho'], polar_coord)))
        y.extend(list(map(lambda x: x['phi'], polar_coord)))

    foreach_dirset(input_csv_file, dir_prefix, process)

    hist = np.histogram2d(x, y, bins=[24, 90])
    norm=matplotlib.colors.LogNorm(vmin=1, vmax=1000, clip=True)
    plt.hist2d(y, x, bins=[90, 60], range=[[-180,180],[0,90]], norm=norm)
    plt.colorbar()
    plt.show()

main()
