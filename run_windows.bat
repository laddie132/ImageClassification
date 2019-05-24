@echo off
rem ---------------------------------------------------------------------------
rem Start script for the Weed Classification Server
rem ---------------------------------------------------------------------------

echo "Starting Weed Classification Server..."
set "HOME_DIR=E:\weed-classification"
set "CREDIT_DIR=E:\2244076_www.cropphoto.cn_other"

cd /d %HOME_DIR%
simple_tensorflow_serving --model_config_file=model_config_file.json --enable_ssl True --secret_pem %CREDIT_DIR%\2244076_www.cropphoto.cn.pem --secret_key %CREDIT_DIR%\2244076_www.cropphoto.cn.key

pause