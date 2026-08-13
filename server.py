import os
import time
import logging
import uuid
import shutil
import asyncio
import csv
from typing import Optional
from enum import Enum
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 引入定时任务调度�?from apscheduler.schedulers.background import BackgroundScheduler

# 调试支持（可选）
if os.getenv("DEBUG") == "1":
    try:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("🐛 Debugpy listening on port 5678")
    except ImportError:
        print("⚠️ DEBUG=1 but debugpy not installed. Run: pip install debugpy")

# --- 引入你的推理�?(保留原来的逻辑) ---
try:
    from rmbg import remove_background
    from musetalk_infer import MuseTalkInference
except ImportError as e:
    print(f"Warning: 依赖库导入失�? {e}")
    # 假桩用于测试
    # MuseTalkInference = None
    # remove_background = None

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 配置目录 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_DIR, "core_data")
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DATA_DIR, "gen_results")
TASKS_CSV_FILE = os.path.join(BASE_DATA_DIR, "tasks.csv") # CSV 文件路径

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 配置参数 ---
RETENTION_SECONDS = 24 * 60 * 60  # 24小时
CLEANUP_INTERVAL_MINUTES = 120     # �?小时检�?
# --- 任务状态定�?---
class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    EXPIRED = "expired"

class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    message: str = ""
    result_url: Optional[str] = None
    created_at: float = 0.0
    timings: Optional[dict] = None

# --- 全局变量 ---
TASKS = {} # 内存缓存，作为快速查询的依据
GPU_LOCK = asyncio.Lock() # GPU 显存�?CSV_LOCK = asyncio.Lock() # CSV 文件写入�?(防止并发写入冲突)
scheduler = BackgroundScheduler()
MUSETALK_ENGINE = None  # 全局模型实例（启动时加载一次）

# --- CSV 持久化工具函�?---

def load_tasks_from_csv():
    """�?CSV 文件加载任务到内�?TASKS 字典"""
    if not os.path.exists(TASKS_CSV_FILE):
        return

    try:
        with open(TASKS_CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 恢复数据类型
                task_id = row['task_id']
                TASKS[task_id] = {
                    "task_id": task_id,
                    "status": row['status'],
                    "created_at": float(row['created_at']),
                    "message": row['message'],
                    "result_path": row['result_path'] if row['result_path'] else None,
                    "result_filename": row['result_filename'] if row['result_filename'] else None,
                    "rm_bg": row['rm_bg'] == 'True' # 字符串转布尔
                }
        logger.info(f"📂 Loaded {len(TASKS)} tasks from CSV.")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")

def save_tasks_to_csv():
    """将内�?TASKS 字典全量写入 CSV 文件"""
    # 注意：在生产环境中，如果数据量巨大，全量重写效率低。但在数字人生成这种低频高耗时场景下完全够用�?    fieldnames = ['task_id', 'status', 'created_at', 'message', 'result_path', 'result_filename', 'rm_bg']
    
    try:
        # 写入临时文件再重命名，防止写入中断导致文件损�?        temp_file = TASKS_CSV_FILE + ".tmp"
        with open(temp_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for task in TASKS.values():
                writer.writerow({
                    'task_id': task['task_id'],
                    'status': task['status'],
                    'created_at': task['created_at'],
                    'message': task.get('message', ''),
                    'result_path': task.get('result_path', ''),
                    'result_filename': task.get('result_filename', ''),
                    'rm_bg': str(task.get('rm_bg', False))
                })
        # 覆盖原文�?        if os.path.exists(TASKS_CSV_FILE):
            os.remove(TASKS_CSV_FILE)
        os.rename(temp_file, TASKS_CSV_FILE)
        # logger.debug("💾 Tasks saved to CSV.")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")

async def async_save_tasks():
    """异步包装的保存函数，用于在接口中使用"""
    async with CSV_LOCK:
        # 在线程池中执行文件IO，避免阻塞事件循�?        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_tasks_to_csv)

def sync_save_tasks_wrapper():
    """同步环境下的保存（如定时任务中）"""
    # 简单的直接保存，不�?async lock (定时任务是独立线�?
    save_tasks_to_csv()


# --- 清理逻辑 ---
def cleanup_expired_files():
    """定时任务：清理过期文件并更新 CSV"""
    logger.info("🧹 Starting scheduled cleanup...")
    now = time.time()
    deleted_count = 0
    expired_tasks = []

    # 1. 找出过期任务
    for task_id, task in TASKS.items():
        if now - task.get("created_at", 0) > RETENTION_SECONDS:
            expired_tasks.append(task_id)
    
    # 2. 从内存移�?    for task_id in expired_tasks:
        del TASKS[task_id]
        logger.info(f"Removed expired task record: {task_id}")
    
    # 3. 物理文件清理 (逻辑保持不变)
    for directory in [UPLOAD_DIR, RESULTS_DIR]:
        if not os.path.exists(directory): continue
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    if now - os.path.getmtime(file_path) > RETENTION_SECONDS:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception: pass
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    if now - os.path.getmtime(dir_path) > RETENTION_SECONDS:
                        shutil.rmtree(dir_path, ignore_errors=True)
                except Exception: pass

    # 4. 如果有变动，保存 CSV
    if expired_tasks:
        sync_save_tasks_wrapper()
        logger.info(f"🧹 Cleanup finished. Records removed: {len(expired_tasks)}, Files deleted: {deleted_count}")

# --- FastAPI 生命周期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MUSETALK_ENGINE

    # 1. 启动时加�?CSV 数据
    load_tasks_from_csv()

    # 2. 启动定时任务
    scheduler.add_job(cleanup_expired_files, 'interval', minutes=CLEANUP_INTERVAL_MINUTES)
    scheduler.start()

    # 3. 预加�?MuseTalk 模型�?GPU（启动时执行一次，所有任务共享）
    logger.info("🔥 Loading MuseTalk models to GPU...")
    loop = asyncio.get_event_loop()
    MUSETALK_ENGINE = await loop.run_in_executor(
        None, MuseTalkInference
    )
    logger.info("🚀 Server started. Models loaded to GPU. Ready for inference.")

    yield

    # 4. 关闭时释放资�?    scheduler.shutdown()
    if MUSETALK_ENGINE is not None:
        del MUSETALK_ENGINE
        logger.info("🧹 GPU memory released.")
    logger.info("🛑 Server shutting down.")

app = FastAPI(title="MuseTalk API with CSV Persistence", lifespan=lifespan)

# --- 核心推理逻辑 ---
def run_musetalk_task(task_id: str, audio_path: str, video_path: str, rm_bg: bool):
    """后台任务：执行推理，并在每一步状态变化时保存 CSV"""

    # 1. 开始处�?    TASKS[task_id]["status"] = TaskStatus.PROCESSING
    save_tasks_to_csv() # 同步写入

    try:
        save_dir = os.path.join(RESULTS_DIR, task_id)
        os.makedirs(save_dir, exist_ok=True)
        result_filename = "musetalk_result.mp4"
        result_path = os.path.join(save_dir, result_filename)

        # 使用全局模型实例（已在启动时加载到GPU�?        _, timings = MUSETALK_ENGINE.generate(
            audio_path=audio_path,
            video_path=video_path,
            result_path=result_path,
            bbox_shift=5, parsing_mode="jaw", extra_margin=8,
            left_cheek_width=60, right_cheek_width=60, batch_size=12
        )
        
        final_output_path = result_path

        if rm_bg:
            rmbg_output_path = result_path.replace(".mp4", "_rmbg.mp4")
            remove_background(result_path, rmbg_output_path, "resnet50")
            final_output_path = rmbg_output_path
            result_filename = os.path.basename(final_output_path)

        # 2. 成功完成
        TASKS[task_id]["status"] = TaskStatus.SUCCESS
        TASKS[task_id]["result_path"] = final_output_path
        TASKS[task_id]["result_filename"] = result_filename
        TASKS[task_id]["timings"] = timings
        logger.info(f"Task {task_id}: Completed")

    except Exception as e:
        # 3. 失败
        logger.error(f"Task {task_id}: Failed: {e}", exc_info=True)
        TASKS[task_id]["status"] = TaskStatus.FAILED
        TASKS[task_id]["message"] = str(e)
    finally:
        # 无论成功失败，最后必须保存状态到 CSV
        save_tasks_to_csv()

async def background_task_wrapper(task_id, audio_path, video_path, rm_bg):
    async with GPU_LOCK:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_musetalk_task, task_id, audio_path, video_path, rm_bg)

# --- API 接口 ---

@app.get("/health")
async def health_check():
    """轻量级健康检查端�?- 用于 Docker HEALTHCHECK"""
    return {
        "status": "ok",
        "service": "musetalk",
        "timestamp": time.time()
    }

@app.get("/readiness")
async def readiness_check():
    """就绪检查端�?- 验证服务完全可用"""
    return {
        "status": "ready",
        "service": "musetalk",
        "tasks_count": len(TASKS),
        "gpu_locked": GPU_LOCK.locked(),
        "timestamp": time.time()
    }

@app.post("/generate", response_model=TaskInfo)
async def create_generation_task(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    video: UploadFile = File(...),
    rm_bg: bool = Form(True)
):
    task_id = str(uuid.uuid4())
    task_upload_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_upload_dir, exist_ok=True)
    
    # 保存文件
    audio_path = os.path.join(task_upload_dir, audio.filename)
    video_path = os.path.join(task_upload_dir, video.filename)
    with open(audio_path, "wb") as b: shutil.copyfileobj(audio.file, b)
    with open(video_path, "wb") as b: shutil.copyfileobj(video.file, b)
        
    # 初始化任�?    TASKS[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "created_at": time.time(),
        "rm_bg": rm_bg,
        "message": "",
        "result_path": "",
        "result_filename": ""
    }
    
    # 立即保存�?CSV，防止服务崩了任务丢�?    await async_save_tasks()
    
    background_tasks.add_task(background_task_wrapper, task_id, audio_path, video_path, rm_bg)
    return TASKS[task_id]

@app.get("/status/{task_id}", response_model=TaskInfo)
async def get_task_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        # 如果内存没有，CSV也没有，那就是真没有或者被清理�?        raise HTTPException(status_code=404, detail="Task not found or expired")

    response = {
        "task_id": task_id,
        "status": task["status"],
        "message": task.get("message", ""),
        "created_at": task.get("created_at")
    }
    
    if task["status"] == TaskStatus.SUCCESS:
        response["result_url"] = f"/download/{task_id}"
        response["timings"] = task.get("timings")

    return response

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    task = TASKS.get(task_id)
    if not task:
         raise HTTPException(status_code=404, detail="Task not found")
         
    if task["status"] != TaskStatus.SUCCESS:
        raise HTTPException(status_code=400, detail="Task not ready")
    
    file_path = task.get("result_path")
    filename = task.get("result_filename", "result.mp4")
    
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="File lost (likely cleaned up)")

    return FileResponse(file_path, media_type="video/mp4", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")
