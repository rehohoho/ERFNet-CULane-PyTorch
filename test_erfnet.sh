
python test_erfnet.py CULane ERFNet train scooter_test_set \
    --gpus='0' \
    --resume='trained/ERFNet_trained.tar' \
    --img_height=208 \
    --img_width=976 \
    --workers=10 \
    --batch-size=5 \
    --lr=0.01