
python test_erfnet.py CULane ERFNet train scooter_test_set \
    --gpus='0' \
    --dataset_path='C:/Users/User/Desktop/ERFNet/list' \
    --resume='trained/ERFNet_trained.tar' \
    --naming_format='.png' \
    --image_height=960 \
    --image_width=1280 \
    --workers=4 \
    --batch-size=2 \
    --lr=0.01

# FOR THE SAD WINDOWS USER:
# python test_erfnet.py CULane ERFNet train scooter_test_set --gpus='0' --resume='trained/ERFNet_trained.tar' --img_height=208 --img_width=976 --workers=10 --batch-size=5 --lr=0.01