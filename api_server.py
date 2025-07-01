import os
import uuid
import torch
import torchaudio
import time
import asyncio
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from pathlib import Path
import shutil
from contextlib import asynccontextmanager

# 导入IndexTTS类
from indextts.infer import IndexTTS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("index-tts-api")

# 目录配置
OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "outputs")
REFERENCE_DIR = os.environ.get("TTS_REFERENCE_DIR", "references")
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "checkpoints")
TEMP_DIR = os.environ.get("TTS_TEMP_DIR", "temp_uploads")
BPE_PATH = os.environ.get("TTS_BPE_PATH", f"{MODEL_DIR}/bpe_cn_en.model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

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

# 定义TTS模型全局实例
tts = None

def initialize_reference_folder():
    """初始化参考音频文件夹结构"""
    if not os.path.exists(REFERENCE_DIR):
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        logger.info(f"创建参考音频文件夹: {REFERENCE_DIR}")

    # 检查是否有示例音频
    if not any(Path(REFERENCE_DIR).glob("*/*.[wm][ap][v3]")):
        logger.warning(
            f"参考音频文件夹为空，请在 {REFERENCE_DIR}/[speaker_id]/ 目录添加.wav或.mp3音频文件")

# 使用异步上下文管理器初始化和管理模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts

    logger.info("初始化IndexTTS模型...")

    # 获取环境变量配置
    compile_mode = False
    fp16_mode = os.environ.get("TTS_FP16", "1") == "1"
    gpu_memory_utilization = float(os.environ.get("TTS_GPU_MEMORY_UTIL", "0.5"))

    # 初始化模型
    tts = IndexTTS(
        model_dir=MODEL_DIR,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        compile=compile_mode and torch.cuda.is_available(),
        is_fp16=fp16_mode and torch.cuda.is_available()
    )

    logger.info("IndexTTS模型初始化完成")

    # 初始化参考音频文件夹
    initialize_reference_folder()

    yield

    # 退出时清理资源
    logger.info("关闭服务，释放资源...")

# 创建应用
app = FastAPI(
    title="IndexTTS API",
    root_path="/tts/api",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "IndexTTS API 服务运行中"}

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/v1/tts", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """异步生成文本到语音转换"""
    task_id = str(uuid.uuid4())

    try:
        # 准备参考音频路径
        reference_ids = request.reference_id.split(",")
        audio_prompts = []

        for ref_id in reference_ids:
            ref_dir = os.path.join(REFERENCE_DIR, ref_id.strip())
            if not os.path.exists(ref_dir):
                raise HTTPException(status_code=404, detail=f"参考音频 {ref_id} 不存在")

            # 查找第一个音频文件
            found_audio = False
            for ext in [".wav", ".mp3"]:
                files = list(Path(ref_dir).glob(f"*{ext}"))
                if files:
                    audio_prompts.append(str(files[0]))
                    found_audio = True
                    break

            if not found_audio:
                raise HTTPException(status_code=404, detail=f"在 {ref_id} 中找不到音频文件")

        if not audio_prompts:
            raise HTTPException(status_code=404, detail="找不到有效的参考音频文件")

        # 准备输出路径
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")

        # 调用推理
        start_time = time.time()

        # 获取全局TTS模型
        global tts
        if tts is None:
            raise HTTPException(status_code=500, detail="TTS模型未初始化")

        # 设置模型参数
        try:
            tts.temperature = request.temperature
            tts.top_p = request.top_p
        except Exception as e:
            logger.warning(f"无法设置temperature或top_p属性: {e}")

        # 在单独的线程中运行TTS推理以避免阻塞事件循环
        if request.stream and hasattr(tts, 'infer_real_stream'):
            # 流式合成
            await asyncio.to_thread(
                tts.infer_real_stream,
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
            # 非流式合成
            await asyncio.to_thread(
                tts.infer_fast,
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

        logger.info(f"任务 {task_id} 完成，耗时 {time.time() - start_time:.2f}秒")

        # 返回结果
        return {
            "id": task_id,
            "audio_url": f"/v1/audio/{task_id}",
            "duration": duration,
            "text": request.text,
            "sampling_rate": info.sample_rate
        }

    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    port = int(os.environ.get("SERVICE_PORT", 8000))
    logger.info(f"启动IndexTTS API服务在端口 {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=120  # 设置keep-alive超时为120秒
    )