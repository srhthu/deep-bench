if [ $1 ] ; then
    gpu=$1
    echo "Set gpu device: $1"
else
    echo "Must specify a device."
    exit 1
fi
CUDA_VISIBLE_DEVICES=${gpu} python run.py \
-n 100 --bs 8 --write