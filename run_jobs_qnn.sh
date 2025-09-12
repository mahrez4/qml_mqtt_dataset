#!/usr/bin/env bash
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Using existing virtual environment: $VENV_DIR"
else
    echo "Creating new virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

python qnn_mqtt_strongentangling.py --backend GPU --fraction 0.01 | tee qnn_mqtt_gpu_strongentangle.txt
python qnn_mqtt_basicentangler.py --backend GPU --fraction 0.01 | tee qnn_mqtt_gpu_basicentangler.txt
python qnn_mqtt_simplifiedtwodesign.py --backend GPU --fraction 0.01 | tee qnn_mqtt_gpu_cat rsimplifiedtwodesign.txt

python qnn_mqtt_strongentangling.py --backend CPU --fraction 0.01 | tee qnn_mqtt_cpu_strongentangle.txt
python qnn_mqtt_basicentangler.py --backend CPU --fraction 0.01 | tee qnn_mqtt_cpu_basicentangler.txt
python qnn_mqtt_simplifiedtwodesign.py --backend CPU --fraction 0.01 | tee qnn_mqtt_cpu_simplifiedtwodesign.txt