# WeedClassification

## Preprocess
```bash
python split_dataset.py --image_dir /home/lh/weed_photos_website --oversampling_num=400 \
--out_image_lists_dir=data/weed_image_lists_mix_oversample.json \
--base_image_lists_dir=data/weed_image_lists.json \
--min_num=100 --dropped_image_lists_dir=data/weed_image_lists_mix_dropped.json
```

## Train

``` bash
python retrain.py --image_dir /home/lh/weed_photos_resize --image_lists_dir data/weed_image_lists_oversample.json \
--output_path outputs/deafult 
```

## Test Metrics
```bash
python test_metrics.py
```

## Rebuild Model
```bash
python rebuild_model.py --origin_model_dir=outputs/weed-mix-sample/saved_model \
--model_dir=outputs/weed-mix-sample/saved_model_b64
```

## Server
```bash
simple_tensorflow_serving --model_config_file=model_config_file.json --enable_ssl True \
--secret_pem E:\2244076_www.cropphoto.cn_other\2244076_www.cropphoto.cn.pem \
--secret_key E:\2244076_www.cropphoto.cn_other\2244076_www.cropphoto.cn.key
```

## Test Client
```bash
python test_client.py
```

## Test Server QPS
```bash
python test_server_qps
```