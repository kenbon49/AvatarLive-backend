import os
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

CheckpointsDir = 'models'

# 下载配置列表
downloads = [
    # MuseTalk v1.0 weights
    ('TMElyralab/MuseTalk', f'{CheckpointsDir}/musetalk', ['musetalk/musetalk.json', 'musetalk/pytorch_model.bin']),
    # MuseTalk v1.5 weights
    ('TMElyralab/MuseTalk', f'{CheckpointsDir}/musetalkV15', ['musetalkV15/musetalk.json', 'musetalkV15/unet.pth']),
    # SD VAE (注意：路径改为 sd-vae 以匹配代码引用)
    ('stabilityai/sd-vae-ft-mse', f'{CheckpointsDir}/sd-vae', ['config.json', 'diffusion_pytorch_model.bin']),
    # Whisper
    ('openai/whisper-tiny', f'{CheckpointsDir}/whisper', ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']),
    # DWPose
    ('yzd-v/DWPose', f'{CheckpointsDir}/dwpose', ['dw-ll_ucoco_384.pth']),
    # SyncNet
    ('ByteDance/LatentSync', f'{CheckpointsDir}/syncnet', ['latentsync_syncnet.pt']),
    # Face Parse Bisent
    ('ManyOtherFunctions/face-parse-bisent', f'{CheckpointsDir}/face-parse-bisent', ['79999_iter.pth', 'resnet18-5c106cde.pth']),
]

for repo_id, local_dir, allow_patterns in downloads:
    print(f"Downloading {repo_id} → {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False  # 禁用符号链接，直接下载
        )
        print(f"✓ Successfully downloaded {repo_id}")
    except Exception as e:
        print(f"✗ Failed to download {repo_id}: {e}")

print("\n✅ All weights download completed!")