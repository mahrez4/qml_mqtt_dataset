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

python qlstm_mqtt.py --backend GPU | tee qlstm_gpu.txt
python qlstm_mqtt.py --backend CPU | tee qlstm_cpu.txt