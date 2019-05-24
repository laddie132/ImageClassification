# WeedClassification

## Train

``` bash
python retrain.py --image_dir /home/lh/weed_photos_resize --image_lists_dir data/weed_image_lists_oversample.json \
--output_path outputs/deafult 
```

## Test Metrics
```bash
python test_metrics.py
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