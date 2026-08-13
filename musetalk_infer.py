import os
import sys
import argparse
import glob
import pickle
import copy
import cv2
import torch
import numpy as np
import imageio
from argparse import Namespace
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import WhisperModel
import shutil
import logging
import time

# --- MuseTalk Imports ---
# 确保在项目根目录运行，否则可能需�?sys.path.append
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MuseTalkInference:
    def __init__(self,
                 model_base_dir="./models",
                 use_float16=False,
                 device=None,
                 version="v15"):
        """
        初始�?MuseTalk 推理引擎，加载模型到内存�?
        Args:
            model_base_dir: 模型存放的根目录
            use_float16: 是否使用半精度推�?(推荐 True，速度快且显存�?
            device: 指定运行设备 'cuda' �?'cpu'，None 则自动检�?            version: 模型版本 'v15' �?'v1' (默认 'v15')
        """
        self.model_base_dir = model_base_dir
        self.use_float16 = use_float16
        self.version = version
        
        # 1. 设备配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"🚀 Initializing MuseTalk {self.version} on {self.device} (FP16={self.use_float16})...")

        # 2. 设置权重精度
        if self.use_float16:
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        # 3. 根据版本选择模型路径
        if self.version == "v15":
            unet_path = os.path.join(model_base_dir, "musetalkV15/unet.pth")
            config_path = os.path.join(model_base_dir, "musetalkV15/musetalk.json")
        elif self.version == "v1":
            unet_path = os.path.join(model_base_dir, "musetalk/musetalk/pytorch_model.bin")
            config_path = os.path.join(model_base_dir, "musetalk/config.json")
        else:
            raise ValueError(f"Unsupported version: {self.version}. Choose 'v15' or 'v1'")

        # 4. 加载核心模型 (VAE, UNet, Positional Encoding)
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_path,
            vae_type="sd-vae",
            unet_config=config_path,
            device=self.device
        )

        # 5. 优化模型精度
        if self.use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.timesteps = torch.tensor([0], device=self.device)

        # 6. 加载 Audio Processor �?Whisper
        logger.info("🔊 Loading Whisper model...")
        self.audio_processor = AudioProcessor(feature_extractor_path=os.path.join(model_base_dir, "whisper"))
        self.whisper = WhisperModel.from_pretrained(os.path.join(model_base_dir, "whisper"))
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        
        logger.info("�?MuseTalk Models Loaded Successfully.")

    @torch.no_grad()
    def generate(self, 
                audio_path: str, 
                video_path: str, 
                result_path: str,
                bbox_shift: int = 5,
                parsing_mode: str = "jaw",
                extra_margin: int = 8,
                left_cheek_width: int = 60,
                right_cheek_width: int = 60,
                batch_size: int = 12
                ) -> tuple:
        """
        执行数字人视频生�?        
        Args:
            audio_path: 驱动音频文件路径 (.wav, .mp3)
            video_path: 参考视频文件路�?(.mp4)
            result_path: 最终结果保存的完整路径 (e.g., /path/to/result.mp4)
            bbox_shift: 边框偏移�?(默认 0)
            parsing_mode: 面部解析模式 'jaw' �?'raw' (默认 'jaw')
            extra_margin: 下巴扩展边缘 (默认 10)
            
        Returns:
            tuple: (result_path, timings) 生成成功的视频路�?+ 各模块耗时字典(�?
        """
        
        # 参数配置封装
        args = Namespace(
            result_dir = os.path.dirname(result_path),
            fps = 25,
            batch_size = batch_size,
            use_saved_coord = False, # 默认不使用缓存坐标，保证稳定�?            audio_padding_length_left = 2,  # 左侧音频填充长度，单位秒（保证视频帧之间的连续新�?            audio_padding_length_right = 2,  # 右侧音频填充长度，单位秒
            version = self.version,  # 使用初始化时指定的版�?            bbox_shift = bbox_shift,
            extra_margin = extra_margin,
            parsing_mode = parsing_mode,
            left_cheek_width = left_cheek_width,
            right_cheek_width = right_cheek_width
        )

        # 检查输入文�?        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 确保输出目录存在
        os.makedirs(args.result_dir, exist_ok=True)
        
        # 预处理视频帧�?(强制转为 25fps 以匹配模�?
        processed_video_path = self._preprocess_video_fps(video_path)
        
        input_basename = os.path.basename(processed_video_path).split('.')[0]
        audio_basename = os.path.basename(audio_path).split('.')[0]
        
        # 临时工作目录
        temp_dir = os.path.join(args.result_dir, "temp_musetalk_" + input_basename + "_" + audio_basename)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 帧提取保存路�?        frames_save_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_save_dir, exist_ok=True)

        try:
            timings = {}
            t_total = time.time()

            t0 = time.time()
            logger.info(f"🎬 Extracting frames from {processed_video_path}...")
            reader = imageio.get_reader(processed_video_path)
            for i, im in enumerate(reader):
                imageio.imwrite(f"{frames_save_dir}/{i:08d}.png", im)
            input_img_list = sorted(glob.glob(os.path.join(frames_save_dir, '*.[jpJP][pnPN]*[gG]')))
            
            timings["frame_extraction"] = time.time() - t0

            # 提取音频特征
            t0 = time.time()
            logger.info("🎵 Extracting audio features...")
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features, 
                self.device, 
                self.weight_dtype, 
                self.whisper, 
                librosa_length,
                fps=args.fps,
                audio_padding_length_left=args.audio_padding_length_left,
                audio_padding_length_right=args.audio_padding_length_right,
            )

            timings["audio_features"] = time.time() - t0

            # 提取人脸关键�?            t0 = time.time()
            logger.info("😊 Extracting face landmarks...")
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, args.bbox_shift)
            
            # 初始�?Face Parsing
            fp = FaceParsing(
                left_cheek_width=args.left_cheek_width,
                right_cheek_width=args.right_cheek_width
            )
            

            timings["face_landmarks"] = time.time() - t0

            # os.makedirs(os.path.join(args.result_dir,'crop_frame'), exist_ok=True)   #后面需要注�?            # counter = 0                                                #后面需要注�?            # 准备推理输入 Latents
            t0 = time.time()
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                y2 = min(y2 + args.extra_margin, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

                # debug_frame_path = os.path.join(args.result_dir,'crop_frame', f"debug_{counter:08d}.png")
                # imageio.imwrite(debug_frame_path, cv2.cvtColor(crop_frame.astype(np.uint8), cv2.COLOR_BGR2RGB))
                # counter += 1

                latents = self.vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
                

            # 循环填充以匹配音频长�?            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            timings["prepare_latents"] = time.time() - t0

            # 批量推理
            t0 = time.time()
            logger.info("�?Starting Batch Inference...")
            video_num = len(whisper_chunks)
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=args.batch_size,
                delay_frame=0,
                device=self.device,
            )
            
            res_frame_list = []
            for _, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num)/args.batch_size)))):
                audio_feature_batch = self.pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self.weight_dtype)
                
                pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = self.vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
                    #将这些图片全都保存到result目录下的temp_dir文件夹中，方便调�?                    
                    # debug_frame_path = os.path.join(args.result_dir,'res_frame', f"debug_{len(res_frame_list):08d}.png")
                    # imageio.imwrite(debug_frame_path, cv2.cvtColor(res_frame.astype(np.uint8), cv2.COLOR_BGR2RGB))


            timings["batch_inference"] = time.time() - t0

            # 合成视频
            t0 = time.time()
            logger.info("🎞�?Blending frames back to video...")
            temp_video_no_audio = os.path.join(temp_dir, "temp_visual.mp4")
            writer = imageio.get_writer(temp_video_no_audio, fps=args.fps, codec='libx264', pixelformat='yuv420p')

            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                y2 = min(y2 + args.extra_margin, ori_frame.shape[0])
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                    combine_frame_rgb = cv2.cvtColor(combine_frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(combine_frame_rgb)
                except Exception as e:
                    logger.info(f"Skipping frame {i} due to error: {e}")
                    continue
            
            writer.close()

            timings["frame_blending"] = time.time() - t0

            # 合并音频
            t0 = time.time()
            logger.info("🔊 Merging audio...")
            video_clip = VideoFileClip(temp_video_no_audio)
            audio_clip = AudioFileClip(audio_path)
            
            # 截断或循环视频以匹配音频长度
            if video_clip.duration > audio_clip.duration:
                video_clip = video_clip.subclip(0, audio_clip.duration)
            
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(result_path, codec='libx264', audio_codec='aac', logger=None)
            
            # 关闭资源
            video_clip.close()
            audio_clip.close()
            reader.close()

            timings["audio_merge"] = time.time() - t0
            timings["total"] = time.time() - t_total

            logger.info(f"�?Finished! Video saved to: {result_path}")
            for k, v in timings.items():
                logger.info(f"   ⏱️  {k}: {v:.2f}s")
            return result_path, timings

        finally:
            # 清理临时文件
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if processed_video_path != video_path and os.path.exists(processed_video_path):
                 os.remove(processed_video_path) # 删除预处理生成的 25fps 视频

    def _preprocess_video_fps(self, video_path, target_fps=25):
        """
        检查视频帧率，如果不是 25fps 则进行转�?        """
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        reader.close()
        
        if abs(fps - target_fps) < 0.1:
            return video_path
            
        logger.info(f"⚠️ Video FPS is {fps}, converting to {target_fps}fps...")
        dir_name = os.path.dirname(video_path)
        base_name = os.path.basename(video_path).split('.')[0]
        new_path = os.path.join(dir_name, f"{base_name}_25fps.mp4")
        
        clip = VideoFileClip(video_path)
        clip.write_videofile(new_path, fps=target_fps, codec='libx264', logger=None)
        clip.close()
        return new_path
    
    
# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化引擎（可选择 v15 �?v1�?    engine = MuseTalkInference(version="v15")  # 使用 v15 版本
    # engine = MuseTalkInference(version="v1")  # 使用 v1 版本
    
    # 2. 定义路径（使用项目内�?data 文件夹）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(base_dir, "data/input/audio/demo2_audio.wav")
    video_file = os.path.join(base_dir, "data/input/video/cy.mp4")
    output_file = os.path.join(base_dir, "data/output/demo2.mp4")

    # 3. 调用生成
    try:
        engine.generate(
            audio_path=audio_file,
            video_path=video_file,
            result_path=output_file,
            bbox_shift=5,
            parsing_mode="jaw",
            extra_margin=8,
            left_cheek_width=60,
            right_cheek_width=60,
            batch_size=12
        )
    except Exception as e:
        print(f"Error: {e}")
