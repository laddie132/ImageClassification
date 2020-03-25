# ImageClassification
Image Classification with TensorFlow, including the preprocessing, training, testing and serving procedure.

## Preprocess
split the dataset to train/dev/test with oversampling and filtering.
```bash
python split_dataset.py \
    --image_dir [IMAGE_PATH] \
    --oversampling_num [OVER_NUM] \
    --out_image_lists_dir [OUT_IMAGE_LISTS_PATH] \
    --min_num [MIN_NUM] \
    --dropped_image_lists_dir [DROPPED_IMAGE_LISTS_DIR]
```

## Train
``` bash
python retrain.py --image_dir [IMAGE_PATH] \
    --image_lists_dir [IMAGE_LISTS_PATH] \
    --output_path [OUT_PATH] \
    --tfhub_module [TFHUB_PATH] \
    --module_trainable 0
```

## Rebuild
replace model input with base64 encoding string.
```bash
python rebuild_model.py --origin_model_dir [IN_MODEL] --model_dir [OUT_MODEL]
```

## Serve
```bash
simple_tensorflow_serving --model_config_file [CONFIG_PATH] \
    [--enable_ssl True --secret_pem [PEM_PATH] --secret_key [KEY_PATH]]
```

## Serve with Flask
serving with TensorFlow frozen model on Flask.
```bash
python flask_server.py --frozen_model_filename [FROZEN_MODEL_PATH]
```

## Test Metrics
```bash
cd tests; python test_metrics.py
```

## Test Client
```bash
cd tests; python test_client.py
```

## Test Server QPS
```bash
cd tests; python test_server_qps.py
```

## Analysis dataset
```bash
cd utils; python dataset_fun.py
```
