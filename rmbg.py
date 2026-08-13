import torch
import os
import logging
from rmbg_model import MattingNetwork
from rmbg_inference import convert_video

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def model_path_mapper(model_name: str) -> str:
    """
    Maps model variant names to their corresponding checkpoint file paths.
    """
    model_paths = {
        'mobilenetv3': './models/rmbg_weight/rvm_mobilenetv3.pth',
        'resnet50': './models/rmbg_weight/rvm_resnet50.pth',
    }
    if model_name not in model_paths:
        raise ValueError(f"Unsupported model variant: {model_name}. Supported variants are: {list(model_paths.keys())}")
    return model_paths[model_name]


def remove_background(input_video_path, output_video_path,model_name='resnet50'):
    model_path = model_path_mapper(model_name)
    model = MattingNetwork(model_name).eval().cuda()  # or "resnet50"  mobilenetv3
    model.load_state_dict(torch.load(model_path))

    output_name = os.path.splitext(os.path.basename(input_video_path))[0]
    alpha_name = output_name + "_alpha.mp4"
    alpha_path = os.path.join(os.path.dirname(output_video_path), alpha_name)
    foreground_name = output_name + "_foreground.mp4"
    foreground_path = os.path.join(os.path.dirname(output_video_path), foreground_name)

    convert_video(
        model,                           # The model, can be on any device (cpu or cuda).
        input_source=input_video_path,        # A video file or an image sequence directory.
        output_type='video',             # Choose "video" or "png_sequence"
        output_composition=output_video_path,    # File path if video; directory path if png sequence.
        output_alpha=alpha_path,          # [Optional] Output the raw alpha prediction.
        output_foreground=foreground_path,     # [Optional] Output the raw foreground prediction.
        output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
        downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
        seq_chunk=12,                    # Process n frames at once for better parallelism.
    )
    # 推理之后释放显卡内存
    torch.cuda.empty_cache()
    return output_video_path


if __name__ == '__main__':

    model_name = 'resnet50'
    input_video_path = r"C:\Users\ddf\Desktop\zzc\code\avatar\test_data\demo2_video.mp4"
    output_video_path = r"C:\Users\ddf\Desktop\zzc\code\avatar\test_data\demo2_video_rmbg.mp4"
    output_video_path = remove_background(input_video_path, output_video_path, model_name=model_name)