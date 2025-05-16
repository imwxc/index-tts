import os
import uuid
import torch
import torchaudio
import time
import asyncio
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import threading
import queue
from pathlib import Path
import shutil

# 导入IndexTTS类
from indextts.infer import IndexTTS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("index-tts-api")

# 定义请求模型
class TTSRequest(BaseModel):
    text: str
    reference_id: str
    temperature: float = 1.0
    top_p: float = 0.8
    speed: float = 1.0
    volume: float = 1.0
    pitch: float = 0.0
    fusion_method: str = "average"
    weights: Optional[List[float]] = None
    no_chunk: bool = False
    stream: bool = False

# 定义响应模型
class TTSResponse(BaseModel):
    id: str
    audio_url: str
    duration: float
    text: str
    sampling_rate: int = 24000

# 创建应用
app = FastAPI(title="IndexTTS API", root_path="/tts/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "outputs")
REFERENCE_DIR = os.environ.get("TTS_REFERENCE_DIR", "references")
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "checkpoints")
TEMP_DIR = os.environ.get("TTS_TEMP_DIR", "temp_uploads")
BPE_PATH = os.environ.get("TTS_BPE_PATH", f"{MODEL_DIR}/bpe_cn_en.model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 工作队列和线程池
task_queue = queue.Queue(maxsize=int(os.environ.get("TTS_MAX_QUEUE", "10")))
results = {}

# 单例模式管理IndexTTS实例
class ModelManager:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("初始化IndexTTS模型...")
                    
                    # 获取环境变量配置
                    compile_mode = os.environ.get("TTS_COMPILE", "1") == "1"
                    fp16_mode = os.environ.get("TTS_FP16", "1") == "1"
                    
                    cls._instance = IndexTTS(
                        model_dir=MODEL_DIR,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        compile=compile_mode and torch.cuda.is_available(),
                        is_fp16=fp16_mode and torch.cuda.is_available()
                    )
                    logger.info("IndexTTS模型初始化完成")
        return cls._instance

def initialize_reference_folder():
    """初始化参考音频文件夹结构"""
    if not os.path.exists(REFERENCE_DIR):
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        logger.info(f"创建参考音频文件夹: {REFERENCE_DIR}")
    
    # 检查是否有示例音频，如果没有可以添加提示日志
    if not any(Path(REFERENCE_DIR).glob("*/*.[wm][ap][v3]")):
        logger.warning(f"参考音频文件夹为空，请在 {REFERENCE_DIR}/[speaker_id]/ 目录添加.wav或.mp3音频文件")

# 工作线程函数
def worker():
    while True:
        try:
            task_id, request = task_queue.get()
            logger.info(f"处理任务 {task_id}")
            
            # 准备参考音频路径
            reference_ids = request.reference_id.split(",")
            audio_prompts = []
            
            for ref_id in reference_ids:
                ref_dir = os.path.join(REFERENCE_DIR, ref_id.strip())
                if not os.path.exists(ref_dir):
                    results[task_id] = {"error": f"参考音频 {ref_id} 不存在"}
                    task_queue.task_done()
                    continue
                
                # 查找第一个音频文件
                for ext in [".wav", ".mp3"]:
                    files = list(Path(ref_dir).glob(f"*{ext}"))
                    if files:
                        audio_prompts.append(str(files[0]))
                        break
            
            if not audio_prompts:
                results[task_id] = {"error": "找不到有效的参考音频文件"}
                task_queue.task_done()
                continue
                
            # 准备输出路径
            output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
            
            # 调用推理
            start_time = time.time()
            try:
                # 获取模型实例
                tts = ModelManager.get_instance()
                
                # 如果IndexTTS类中有可设置的属性，可以在这里设置
                # 注意：这部分取决于IndexTTS的实际实现方式，可能不需要
                # 如果这些属性不存在，可能会引发错误
                try:
                    tts.temperature = request.temperature
                    tts.top_p = request.top_p
                except:
                    logger.warning("无法设置temperature或top_p属性，这些参数可能不会生效")
                
                # 调用infer_fast方法
                if request.stream and hasattr(tts, 'infer_real_stream'):
                    # 流式合成
                    tts.infer_real_stream(
                        audio_prompt=audio_prompts,
                        text=request.text,
                        output_path=output_path,
                        verbose=False,
                        prompt_id=request.reference_id,
                        fusion_method=request.fusion_method,
                        weights=request.weights,
                        buffer_size=25
                    )
                else:
                    # 非流式合成 - 按照实际参数列表调整
                    tts.infer_fast(
                        audio_prompt=audio_prompts,
                        text=request.text,
                        output_path=output_path,
                        verbose=False,
                        prompt_id=request.reference_id,
                        fusion_method=request.fusion_method,
                        weights=request.weights,
                        no_chunk=request.no_chunk
                    )
                
                # 计算音频时长
                info = torchaudio.info(output_path)
                duration = info.num_frames / info.sample_rate
                
                # 存储结果
                results[task_id] = {
                    "id": task_id,
                    "audio_url": f"/v1/audio/{task_id}",
                    "duration": duration,
                    "text": request.text,
                    "sampling_rate": info.sample_rate
                }
                
                logger.info(f"任务 {task_id} 完成，耗时 {time.time() - start_time:.2f}秒")
            
            except Exception as e:
                logger.error(f"任务 {task_id} 失败: {str(e)}")
                results[task_id] = {"error": str(e)}
            
            task_queue.task_done()
        
        except Exception as e:
            logger.error(f"工作线程出错: {str(e)}")

# 上传文件处理函数
async def process_upload_task(task_id: str, text: str, audio_file_path: str, 
                              temperature: float = 1.0, top_p: float = 0.8, 
                              fusion_method: str = "average", no_chunk: bool = False):
    """处理上传的音频文件和文本生成语音"""
    try:
        # 准备音频文件路径
        audio_prompts = [audio_file_path]
                
        # 准备输出路径
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
        
        # 调用推理
        start_time = time.time()
        
        # 获取模型实例
        tts = ModelManager.get_instance()
        
        # 设置模型参数
        try:
            tts.temperature = temperature
            tts.top_p = top_p
        except:
            logger.warning("无法设置temperature或top_p属性，这些参数可能不会生效")
        
        # 调用推理
        tts.infer_fast(
            audio_prompt=audio_prompts,
            text=text,
            output_path=output_path,
            verbose=False,
            prompt_id="uploaded_audio",
            fusion_method=fusion_method,
            weights=None,
            no_chunk=no_chunk
        )
        
        # 计算音频时长
        info = torchaudio.info(output_path)
        duration = info.num_frames / info.sample_rate
        
        # 存储结果
        results[task_id] = {
            "id": task_id,
            "audio_url": f"/v1/audio/{task_id}",
            "duration": duration,
            "text": text,
            "sampling_rate": info.sample_rate
        }
        
        logger.info(f"上传任务 {task_id} 完成，耗时 {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"上传任务 {task_id} 失败: {str(e)}")
        results[task_id] = {"error": str(e)}
    
    finally:
        # 清理临时文件
        try:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logger.debug(f"临时文件已删除: {audio_file_path}")
        except Exception as e:
            logger.warning(f"删除临时文件失败: {str(e)}")

# 启动工作线程
initialize_reference_folder()
threading.Thread(target=worker, daemon=True).start()

@app.get("/")
def root():
    return {"message": "IndexTTS API 服务运行中"}

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/v1/tts", response_model=TTSResponse)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """生成文本到语音转换"""
    task_id = str(uuid.uuid4())
    
    try:
        # 将任务添加到队列
        task_queue.put((task_id, request))
        logger.info(f"任务 {task_id} 添加到队列")
        
        # 等待任务完成，设置超时为60秒
        timeout = 120  # 超时时间（秒）
        max_iterations = int(timeout / 0.1)  # 每0.1秒检查一次
        
        for _ in range(max_iterations):  
            if task_id in results:
                if "error" in results[task_id]:
                    error_msg = results[task_id]["error"]
                    del results[task_id]
                    raise HTTPException(status_code=500, detail=error_msg)
                
                result = results[task_id]
                # 创建后台任务在一段时间后清理结果
                background_tasks.add_task(lambda: results.pop(task_id, None))
                return result
            
            # 等待100毫秒
            await asyncio.sleep(0.1)
        
        raise HTTPException(status_code=408, detail="处理超时 (60秒)")
    
    except queue.Full:
        raise HTTPException(status_code=503, detail="服务器队列已满，请稍后再试")


@app.get("/v1/audio/{audio_id}")
async def get_audio(audio_id: str):
    """获取生成的音频文件"""
    file_path = os.path.join(OUTPUT_DIR, f"{audio_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")
    
    return FileResponse(
        file_path, 
        media_type="audio/wav", 
        filename=f"{audio_id}.wav"
    )

@app.post("/v1/tts_audio")
async def generate_and_return_tts_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """生成文本到语音并直接返回音频文件"""
    task_id = str(uuid.uuid4())
    
    try:
        # 将任务添加到队列
        task_queue.put((task_id, request))
        logger.info(f"任务 {task_id} 添加到队列")
        
        # 等待任务完成，设置超时为60秒
        timeout = 60  # 超时时间（秒）
        max_iterations = int(timeout / 0.1)  # 每0.1秒检查一次
        
        for _ in range(max_iterations):  
            if task_id in results:
                if "error" in results[task_id]:
                    error_msg = results[task_id]["error"]
                    del results[task_id]
                    raise HTTPException(status_code=500, detail=error_msg)
                
                # 任务已完成，直接返回音频文件
                file_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail="音频文件不存在")
                
                # 创建后台任务在一段时间后清理结果
                background_tasks.add_task(lambda: results.pop(task_id, None))
                
                return FileResponse(
                    file_path, 
                    media_type="audio/wav", 
                    filename=f"{task_id}.wav"
                )
            
            # 等待100毫秒
            await asyncio.sleep(0.1)
        
        raise HTTPException(status_code=408, detail="处理超时 (60秒)")
    
    except queue.Full:
        raise HTTPException(status_code=503, detail="服务器队列已满，请稍后再试")

@app.get("/v1/references")
async def list_references():
    """列出所有可用的参考音频ID"""
    try:
        if not os.path.exists(REFERENCE_DIR):
            return {"references": []}
            
        references = []
        for item in Path(REFERENCE_DIR).iterdir():
            if item.is_dir():
                # 检查目录中是否有音频文件
                has_audio = False
                for file in item.iterdir():
                    if file.suffix.lower() in ['.wav', '.mp3']:
                        has_audio = True
                        break
                        
                if has_audio:
                    references.append({
                        "id": item.name,
                        "name": item.name
                    })
                
        return {"references": references}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取参考音频列表失败: {str(e)}")

# 流式TTS接口
@app.post("/v1/tts_stream")
async def stream_tts(request: TTSRequest):
    """流式TTS合成 (仅返回完整音频，未实现真正的流式响应)"""
    # 使用与普通TTS相同的方式处理，但设置stream标志为True
    request.stream = True
    return await generate_tts(request, BackgroundTasks())

if __name__ == "__main__":
    port = int(os.environ.get("SERVICE_PORT", 8000))
    logger.info(f"启动IndexTTS API服务在端口 {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=60  # 设置keep-alive超时为60秒
    )