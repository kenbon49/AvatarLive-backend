from rmbg import remove_background
from musetalk_infer import MuseTalkInference
import logging
import os
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_data_dir():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(data_dir,'gen_results')
    return data_dir


def musetalk_infer(audio_path, 
                video_path, 
                result_path = None, 
                rm_bg=True,
                rmbg_model_name="resnet50",
                bbox_shift=5, 
                parsing_mode="jaw", 
                extra_margin=8, 
                left_cheek_width=60, 
                right_cheek_width=60, 
                batch_size=12):
    
    base_result_dir = get_data_dir()
    save_dir = os.path.join(base_result_dir, f"{time.strftime('%Y%m%d_%H%M%S')}-musetalk")
    if result_path == None:
        result_path = os.path.join(save_dir, "musetalk_result.mp4")
        
    engine = MuseTalkInference()
    engine.generate(
        audio_path=audio_path,
        video_path=video_path,
        result_path=result_path,
        bbox_shift=bbox_shift,
        parsing_mode=parsing_mode,
        extra_margin=extra_margin,
        left_cheek_width=left_cheek_width,
        right_cheek_width=right_cheek_width,
        batch_size=batch_size
    )
    logger .info(f"âœ?MuseTalk Inference completed! Result saved at: {result_path}")

    if rm_bg:
        rmbg_output_path = result_path.replace(".mp4", "_rmbg.mp4")
        logger.info("ðŸª„ Removing background using RMBG...")
        remove_background(
            input_video_path=result_path,
            output_video_path=rmbg_output_path,
            model_name=rmbg_model_name
        )
        logger.info(f"âœ?Background removal completed! Result saved at: {rmbg_output_path}")
        return rmbg_output_path
    else:
        return result_path

if __name__ == "__main__":
    audio_path = r"C:\Users\ddf\Desktop\zzc\code\avatar\test_data\demo2_audio.wav"
    video_path = r"C:\Users\ddf\Desktop\zzc\code\avatar\test_data\demo2_video.mp4"
    result_path = r"C:\Users\ddf\Desktop\zzc\code\avatar\test_data\demo2_result.mp4"
    rm_bg = True
    result_path = musetalk_infer(
        audio_path=audio_path,
        video_path=video_path,
        result_path=result_path,
        rm_bg=rm_bg,
    )
