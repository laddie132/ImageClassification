#!/usr/bin/env bash

SUB_PATH=weed-inception-inception_resnet
IMAGE_LIST_NAME=weed_image_lists_mix_oversample.json
IMAGE_PATH_NAME=weed_photos_resize_mix
TFHUB_MODULE="hub_modules/inception-v3-feature"\ "hub_modules/inception_resnet_v2"

CUDA_VISIBLE_DEVICES=0 python retrain.py \
    --image_dir /home/lh/$IMAGE_PATH_NAME \
    --image_lists_dir data/$IMAGE_LIST_NAME \
    --output_path outputs/${SUB_PATH} \
    --tfhub_module $TFHUB_MODULE \
    --module_trainable 0

python rebuild_model.py \
    --model_dir outputs/${SUB_PATH}/saved_model_b64 \
    --origin_model_dir outputs/${SUB_PATH}/saved_model
