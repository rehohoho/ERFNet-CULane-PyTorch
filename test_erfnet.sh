
python test_erfnet.py CULane ERFNet train tusimple_0531 \
    --gpus='0' \
    --dataset_path='C:/Users/User/Desktop/ERFNet/list' \
    --resume='trained/ERFNet_trained.tar' \
    --naming_format='.jpg' \
    --height_from_bottom=0.7 \
    --image_height=720 \
    --image_width=1280 \
    --workers=4 \
    --batch_size=2 \
    --lr=0.01