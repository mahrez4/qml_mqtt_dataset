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

python qsvm_mqtt.py --backend GPU --encoding angle --fraction 0.1 > qsvm_GPU_result.txt
python qsvm_mqtt.py --backend CPU --encoding angle --fraction 0.1 > qsvm_CPU_result.txt