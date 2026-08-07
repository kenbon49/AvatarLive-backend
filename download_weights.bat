@echo off
setlocal

:: Set the checkpoints directory
set CheckpointsDir=models

:: Create necessary directories
mkdir %CheckpointsDir%\musetalk 2>nul
mkdir %CheckpointsDir%\musetalkV15 2>nul
mkdir %CheckpointsDir%\syncnet 2>nul
mkdir %CheckpointsDir%\dwpose 2>nul
mkdir %CheckpointsDir%\face-parse-bisent 2>nul
mkdir %CheckpointsDir%\sd-vae 2>nul
mkdir %CheckpointsDir%\whisper 2>nul

:: Install required packages
pip install -U "huggingface_hub[cli]"

:: Set HuggingFace endpoint
set HF_ENDPOINT=https://hf-mirror.com

echo Downloading MuseTalk v1.0 weights...
hf download TMElyralab/MuseTalk --local-dir %CheckpointsDir%\musetalk --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"

echo Downloading MuseTalk v1.5 weights...
hf download TMElyralab/MuseTalk --local-dir %CheckpointsDir%\musetalkV15 --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

echo Downloading SD VAE weights...
hf download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae --include "config.json" "diffusion_pytorch_model.bin"

echo Downloading Whisper weights...
hf download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

echo Downloading DWPose weights...
hf download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose --include "dw-ll_ucoco_384.pth"

echo Downloading SyncNet weights...
hf download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet --include "latentsync_syncnet.pt"

echo Downloading Face Parse Bisent weights...
hf download ManyOtherFunctions/face-parse-bisent --local-dir %CheckpointsDir%\face-parse-bisent --include "79999_iter.pth" "resnet18-5c106cde.pth"

echo.
echo ✅ All weights have been downloaded successfully!
endlocal 
