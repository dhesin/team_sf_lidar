bag=$(basename $1)
dirset=$(dirname $1)
python extract_rosbag.py --quiet --pickle --outdir $PROCESSED_DATA_HOME/$1 --interpolate $PROCESSED_DATA_HOME/$1/tracklet_labels.xml $PROCESSED_DATA_HOME/$1/$bag.bag --camera_calibration ../data/calibration/camera_calibration.yaml
