@echo off

echo clone models
git lfs install
git clone https://huggingface.co/tk93/V-Express
move V-Express\model_ckpts model_ckpts

echo Install Depends
python -m venv venv
call venv/scripts/activate
pip install -r requerements.txt

echo Install GPU libs
pip install torch==2.0.1+cu118 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu

echo install complete
pause