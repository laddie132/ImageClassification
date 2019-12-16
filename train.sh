#!/usr/bin/env bash

SUB_PATH=weed-v4-inaturalist-inception
IMAGE_LIST_NAME=v4/weed_image_lists_oversample.json
IMAGE_PATH_NAME=weed_photos_resize_v4
TFHUB_MODULE="hub_modules/inaturalist-inception_v3"

CUDA_VISIBLE_DEVICES=3 python retrain.py \
    --image_dir /home/lh/$IMAGE_PATH_NAME \
    --image_lists_dir data/$IMAGE_LIST_NAME \
    --output_path outputs/${SUB_PATH} \
    --tfhub_module $TFHUB_MODULE \
    --module_trainable 0

CUDA_VISIBLE_DEVICES=3 python rebuild_model.py \
    --model_dir outputs/${SUB_PATH}/saved_model_b64 \
    --origin_model_dir outputs/${SUB_PATH}/saved_model
