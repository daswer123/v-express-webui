#!/bin/bash

echo "Clone models"
git lfs install
git clone https://huggingface.co/tk93/V-Express
mv V-Express/model_ckpts model_ckpts

echo "Install dependencies"
python3 -m venv venv
source venv/bin/activate
pip install -r requerements.txt

echo "Install GPU libraries"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

echo "Installation complete"
read -p "Press any key to continue..."