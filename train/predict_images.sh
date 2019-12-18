output=$1
model=$2
predict_csv=$3
python predict.py $model $predict_csv --export --dir_prefix $PROCESSED_DATA_HOME --output_dir $PROCESSED_DATA_HOME/$1