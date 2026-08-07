import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, feature_extractor_path="openai/whisper-tiny/"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path, start_index=0, weight_dtype=None):
        if not os.path.exists(wav_path):
            return None
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000
        # Split audio into 30s segments
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i:i + segment_length] for i in range(0, len(librosa_output), segment_length)]

        features = []
        for segment in segments:
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        return features, len(librosa_output)

    def get_whisper_chunk(
        self,
        whisper_input_features,    # 裁剪后的30s 梅尔频谱特征列表 
        device,
        weight_dtype,
        whisper,                   # Whisper 模型
        librosa_length,              # 原始音频长度，单位采样点
        fps=25,
        audio_padding_length_left=2,  # 左侧音频填充长度，单位秒（保证视频帧之间的连续新）
        audio_padding_length_right=2,  # 右侧音频填充长度，单位秒
    ):
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1) #可以查看whisper的梅尔特征帧来计算
        whisper_feature = []
        # Process multiple 30s mel input features
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states # 12层隐藏状态
            audio_feats = torch.stack(audio_feats, dim=2) # 合并所有层的隐藏状态
            whisper_feature.append(audio_feats) 

        whisper_feature = torch.cat(whisper_feature, dim=1)
        # Trim the last segment to remove padding
        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:,:actual_length,...]  # 去掉填充部分的音频特征

        # Calculate padding amount
        padding_nums = math.ceil(whisper_idx_multiplier)
        # Add padding at start and end
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),
            whisper_feature,
            # Add extra padding to prevent out of bounds
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right])
        ], 1)   #填充空音频特征

        audio_prompts = []
        for frame_index in range(num_frames):
            try:
                audio_index = math.floor(frame_index * whisper_idx_multiplier)    # 计算当前视频帧对应的音频索引
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                assert audio_clip.shape[1] == audio_feature_length_per_frame
                audio_prompts.append(audio_clip)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}-{audio_index + audio_feature_length_per_frame}")
                exit()

        audio_prompts = torch.cat(audio_prompts, dim=0)  # T, 10, 5, 384
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')
        return audio_prompts

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    wav_path = "./2.wav"
    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(wav_path)
    print("Audio Feature shape:", audio_feature.shape)
    print("librosa_feature_length:", librosa_feature_length)

