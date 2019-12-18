bag=$(basename $1)
dirset=$(dirname $1)
model=$2
python predict.py --export --dir_prefix $PROCESSED_DATA_HOME --output_dir $PROCESSED_DATA_HOME/$1 --data_type rosbag --weightsFile $3 $2 $PROCESSED_DATA_HOME/$1/$bag.bag