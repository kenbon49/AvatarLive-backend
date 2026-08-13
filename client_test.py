import requests
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AVATAR_MUSETALK_URL = "http://10.4.124.27:8083/generate"



def get_data_dir():
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(data_dir,'data')
    return data_dir


class MuseTalkClient:
    def __init__(self):
        """
        初始化客户端
        :param base_url: 服务端地址，例如 "http://10.4.124.27:8083"
        """
        self.base_url = "http://10.4.124.27:8083"
    
    def submit_task(self, audio_path, video_path, rm_bg=False):
        """
        提交生成任务
        :param audio_path: 音频文件路径
        :param video_path: 视频文件路径
        :param rm_bg: 是否去背景
        :return: 任务ID
        """
        files = [
            ('audio', ('audio.wav', open(audio_path, 'rb'), 'audio/wav')),
            ('video', ('video.mp4', open(video_path, 'rb'), 'video/mp4'))
        ]
        payload = {'rm_bg': str(rm_bg)}
        response = requests.post(f"{self.base_url}/generate", files=files, data=payload)
        if response.status_code != 200:
            logger.error(f"任务提交失败: {response.text}")
            raise Exception("任务提交失败")
        
        data = response.json()
        logger.info(f"提交任务 {data}")

        return data['task_id']
    
    def check_status(self, task_id):
        """
        查询任务状态
        :param task_id: 任务ID
        :return: 任务状态
        """
        status_url = f"{self.base_url}/status/{task_id}"
        response = requests.get(status_url)
        if response.status_code != 200:
            logger.error(f"查询任务状态失败: {response.text}")
            raise Exception("查询任务状态失败")
        
        data = response.json()
        logger.info(f"查询任务状态 {data}")

        return data['status'], data.get('result_url'), data.get('message'), data.get('timings')
    

    def download_result(self, result_url, save_path = None, task_id = None):
        """
        下载生成结果
        :param result_url: 结果下载链接
        :param save_path: 保存路径
        :param task_id: 任务ID，用于生成默认文件名
        """
        data_dir = get_data_dir()

        if save_path is None:
            save_dir = os.path.join(data_dir,"result_video","avatar_video",)
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{task_id}.mp4" if task_id else f"{int(time.time())}.mp4"
            save_path = os.path.join(save_dir, fname)
        
        download_url = f"{self.base_url}{result_url}"
        response = requests.get(download_url)
        if response.status_code != 200:
            logger.error(f"下载结果失败: {response.text}")
            raise Exception("下载结果失败")
        with open(save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"结果已保存到 {save_path}")


if __name__ == "__main__":

    audio_path = r'D:\code\avatar\MuseTalk\data\input\audio\demo2_audio.wav'
    video_path = r'D:\temp\heygen.mp4'
    client = MuseTalkClient()

    # ===== 阶段 1：上传计时 =====
    t_upload_start = time.perf_counter()
    task_id = client.submit_task(audio_path, video_path, rm_bg=False)
    t_upload_done = time.perf_counter()
    upload_duration = t_upload_done - t_upload_start

    logger.info(f"任务已提交，ID: {task_id}")
    logger.info(f"⏱️ 上传耗时: {upload_duration:.3f} 秒")

    # ===== 阶段 2：推理等待计时 =====
    # 注意：客户端无法精确知道服务端何时真正完成推理，
    # 这里测量的是「上传完成 → 第一次看到 success」的耗时，
    # 实际推理耗时 ≤ 此值，误差最多为一个轮询间隔(5秒)。
    poll_count = 0
    while True:
        status, result_url, message, timings = client.check_status(task_id)
        poll_count += 1
        logger.info(f"当前状态: {status}")

        if status == 'success':
            t_inference_done = time.perf_counter()
            inference_wait = t_inference_done - t_upload_done

            client.download_result(result_url, task_id=task_id)
            t_all_done = time.perf_counter()
            download_duration = t_all_done - t_inference_done
            total_duration = t_all_done - t_upload_start

            logger.info("生成成功！结果已下载。")
            logger.info("=" * 55)
            logger.info("⏱️  客户端统计 (含轮询误差 ≤5秒)")
            logger.info(f"  上传耗时:       {upload_duration:.3f} 秒")
            logger.info(f"  推理等待耗时:   {inference_wait:.3f} 秒")
            logger.info(f"  下载耗时:       {download_duration:.3f} 秒")
            logger.info(f"  端到端总耗时:   {total_duration:.3f} 秒")
            logger.info(f"  轮询次数:       {poll_count}")
            if timings:
                logger.info("-" * 55)
                logger.info("⏱️  服务端各模块推理耗时 (精确)")
                for phase, t in timings.items():
                    logger.info(f"  {phase:>20s}: {t:.2f}s")
            logger.info("=" * 55)
            break
        elif status == 'failed':
            logger.error(f"生成失败: {message}")
            break

        time.sleep(5)
        
        
    