
ERFNET_FOLDER="C:/Users/User/Desktop/ERFNet"

python get_lane.py \
    --data_root="${ERFNET_FOLDER}/list/" \
    --prob_root="${ERFNET_FOLDER}/predicts/vgg_SCNN_DULR_w9" \
    --output_root="${ERFNET_FOLDER}/tools/output/vgg_SCNN_DULR_w9" \
    --test_list="${ERFNET_FOLDER}/list/scooter_test_set.txt"