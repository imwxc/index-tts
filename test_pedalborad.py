# pylint: disable=all

import numpy as np
from pedalboard import Pedalboard, Chorus, Reverb, Phaser, Delay, Compressor
from pedalboard.io import AudioFile
import sounddevice as sd
import soundfile as sf
import os
import gradio as gr
import tempfile
from datetime import datetime

audio_path = "outputs/results/qianqiuhao_clone/gen_脚本_101_text.wav"


def mono_to_stereo_pedalboard(
    input_file, 
    output_file=None,
    # 左声道参数
    left_chorus_rate=0.1,
    left_chorus_depth=0.2,
    left_chorus_mix=0.7,
    left_reverb_room_size=0.05,
    left_reverb_wet_level=0.2,
    # 右声道参数
    right_phaser_rate=0.1,
    right_phaser_depth=0.2,
    right_phaser_mix=0.7,
    right_delay_seconds=0.05,
    right_delay_feedback=0.1,
    right_delay_mix=0.15,
    right_reverb_room_size=0.05,
    right_reverb_wet_level=0.2,
    # 主压缩器参数
    master_threshold_db=-20,
    master_ratio=4
):
    """
    将单声道音频文件转换为立体声并应用效果器

    参数:
    input_file -- 输入的音频文件路径或音频数据
    output_file -- 输出的音频文件路径，如为None则不保存文件
    左右声道和主压缩器的各种参数
    """
    # 输出调试信息
    print(f"输入类型: {type(input_file)}")
    
    # 根据输入类型处理
    if isinstance(input_file, str):
        print(f"处理文件: {input_file}")

        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 输入文件 {input_file} 不存在")
            return False
        
        # 读取音频文件
        with AudioFile(input_file) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate
    elif isinstance(input_file, tuple):
        # 假设输入是Gradio的音频格式 (sr, audio) 或 (audio, sr)
        print(f"从元组获取音频数据: 长度={len(input_file)}")
        
        if len(input_file) != 2:
            print(f"错误: 元组长度不是2，而是 {len(input_file)}")
            return False
            
        # 尝试判断哪个是音频数据，哪个是采样率
        if isinstance(input_file[0], (int, float)) and isinstance(input_file[1], np.ndarray):
            # 格式是 (sr, audio)
            sample_rate = int(input_file[0])
            audio = input_file[1]
            print(f"检测到格式: (采样率, 音频数据)")
        elif isinstance(input_file[1], (int, float)) and isinstance(input_file[0], np.ndarray):
            # 格式是 (audio, sr)
            audio = input_file[0]
            sample_rate = int(input_file[1])
            print(f"检测到格式: (音频数据, 采样率)")
        else:
            print(f"错误: 无法解析元组数据类型: ({type(input_file[0])}, {type(input_file[1])})")
            return False
    else:
        print(f"不支持的输入类型: {type(input_file)}")
        return False

    # 检查音频数据是否有效
    if not isinstance(audio, np.ndarray):
        print(f"错误: 音频数据不是numpy数组，而是 {type(audio)}")
        return False
    
    # 检查音频数据类型，确保是浮点型
    if not np.issubdtype(audio.dtype, np.floating):
        print(f"将音频数据从 {audio.dtype} 转换为 float32")
        # 如果是整数类型，根据位深度进行归一化
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / max(abs(info.min), info.max)
        else:
            # 其他类型直接转换为浮点型
            audio = audio.astype(np.float32)
    
    print(f"音频数据类型: {audio.dtype}, 形状: {audio.shape}")
    
    if audio.size == 0:
        print("错误: 读取的音频数据为空")
        return False

    print(f"原始音频形状: {audio.shape}, 采样率: {sample_rate}")

    # 确定音频形状和通道数
    if len(audio.shape) == 1:
        # 单维数组：形状为 (n_samples,)
        print("检测到单维数组音频")
        mono_audio = audio
        is_mono = True
    elif len(audio.shape) == 2:
        if audio.shape[0] == 1:
            # 形状为 (1, n_samples)，可能是单声道
            print("检测到形状为 (1, n_samples) 的音频")
            mono_audio = audio[0]  # 转为一维数组
            is_mono = True
        elif audio.shape[1] == 1:
            # 形状为 (n_samples, 1)，单声道
            print("检测到形状为 (n_samples, 1) 的音频")
            mono_audio = audio.flatten()
            is_mono = True
        elif audio.shape[1] == 2:
            # 形状为 (n_samples, 2)，立体声
            print("检测到形状为 (n_samples, 2) 的立体声音频")
            stereo_audio = audio
            is_mono = False
        else:
            # 其他形状，可能是多声道
            print(f"检测到不常见的音频形状: {audio.shape}")
            if audio.shape[0] == 2:
                # 可能是 (2, n_samples)，尝试转置
                print("尝试将形状为 (2, n_samples) 的音频转置为 (n_samples, 2)")
                stereo_audio = audio.T
                is_mono = False
            else:
                # 不确定如何处理，默认使用第一个通道作为单声道
                print("使用第一个通道作为单声道")
                mono_audio = audio[:, 0]
                is_mono = True
    else:
        # 更复杂的形状，不确定如何处理
        print(f"无法处理的音频形状: {audio.shape}")
        return False

    # 如果是单声道，转换为立体声
    if is_mono:
        stereo_audio = np.column_stack((mono_audio, mono_audio))
        print(f"单声道转换为立体声，形状: {stereo_audio.shape}")

    # 创建左右声道的效果器，使用传入的参数
    left_board = Pedalboard(
        [
            Chorus(rate_hz=left_chorus_rate, depth=left_chorus_depth, mix=left_chorus_mix),
            Reverb(room_size=left_reverb_room_size, wet_level=left_reverb_wet_level),
        ]
    )

    right_board = Pedalboard(
        [
            Phaser(rate_hz=right_phaser_rate, depth=right_phaser_depth, mix=right_phaser_mix),
            Delay(delay_seconds=right_delay_seconds, feedback=right_delay_feedback, mix=right_delay_mix),
            Reverb(room_size=right_reverb_room_size, wet_level=right_reverb_wet_level),
        ]
    )

    # 处理左右声道
    left_channel = left_board(stereo_audio[:, 0], sample_rate)
    right_channel = right_board(stereo_audio[:, 1], sample_rate)

    # 将左右声道合并为一个新的立体声音频
    processed_audio = np.column_stack((left_channel, right_channel))
    
    print(f"效果器处理后音频形状: {processed_audio.shape}")

    # 添加一个压缩器使整体音量一致
    master_board = Pedalboard(
        [Compressor(threshold_db=master_threshold_db, ratio=master_ratio, attack_ms=5, release_ms=100)]
    )
    processed_audio = master_board(processed_audio, sample_rate)

    print(f"最终处理后音频形状: {processed_audio.shape}")

    # 如果提供了输出文件路径，则保存文件
    if output_file:
        sf.write(output_file, processed_audio, sample_rate)
        print(f"已保存处理后的音频到: {output_file}")

    # 返回处理后的音频和采样率
    return (processed_audio, sample_rate)


def play_audio(audio_file):
    """
    使用sounddevice播放音频文件

    参数:
    audio_file -- 要播放的音频文件路径
    """
    try:
        data, sample_rate = sf.read(audio_file)
        print(f"播放音频: {audio_file}, 形状: {data.shape}, 采样率: {sample_rate}")

        # 播放音频
        sd.play(data, sample_rate)
        # 等待音频播放完成
        sd.wait()
        print(f"音频播放完成: {audio_file}")
    except Exception as e:
        print(f"播放音频时发生错误: {e}")


def process_audio_for_gradio(
    audio_input, 
    # 左声道参数
    left_chorus_rate,
    left_chorus_depth,
    left_chorus_mix,
    left_reverb_room_size,
    left_reverb_wet_level,
    # 右声道参数
    right_phaser_rate,
    right_phaser_depth,
    right_phaser_mix,
    right_delay_seconds,
    right_delay_feedback,
    right_delay_mix,
    right_reverb_room_size,
    right_reverb_wet_level,
    # 主压缩器参数
    master_threshold_db,
    master_ratio
):
    """处理音频并返回Gradio可接受的格式"""
    if audio_input is None:
        return None, "请先上传音频文件"
        
    try:
        # 调试信息：检查输入类型
        print(f"Gradio音频输入: {audio_input}")
        print(f"Gradio音频输入类型: {type(audio_input)}")
        
        # 处理音频
        processed_result = mono_to_stereo_pedalboard(
            audio_input,
            None,  # 不保存到文件
            left_chorus_rate,
            left_chorus_depth,
            left_chorus_mix,
            left_reverb_room_size,
            left_reverb_wet_level,
            right_phaser_rate,
            right_phaser_depth,
            right_phaser_mix,
            right_delay_seconds,
            right_delay_feedback,
            right_delay_mix,
            right_reverb_room_size,
            right_reverb_wet_level,
            master_threshold_db,
            master_ratio
        )
        
        if processed_result is False:
            return None, "音频处理失败"
            
        processed_audio, sample_rate = processed_result
        
        # 创建临时文件以供Gradio播放
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, processed_audio, sample_rate)
        temp_file.close()
        
        return temp_file.name, "音频处理成功"
    except Exception as e:
        import traceback
        error_info = traceback.format_exc()
        return None, f"处理音频时发生错误: {e}\n{error_info}"


def apply_preset(preset_name):
    """应用预设效果，返回相应的参数值"""
    if preset_name == "默认效果":
        return [
            0.1, 0.2, 0.7, 0.05, 0.2,  # 左声道
            0.1, 0.2, 0.7, 0.05, 0.1, 0.15, 0.05, 0.2,  # 右声道
            -20, 4  # 主控
        ]
    elif preset_name == "强混响效果":
        return [
            0.1, 0.3, 0.8, 0.8, 0.6,  # 左声道
            0.1, 0.3, 0.8, 0.1, 0.2, 0.3, 0.8, 0.6,  # 右声道
            -30, 6  # 主控
        ]
    elif preset_name == "轻微效果":
        return [
            0.05, 0.1, 0.3, 0.02, 0.1,  # 左声道
            0.05, 0.1, 0.3, 0.03, 0.05, 0.08, 0.02, 0.1,  # 右声道
            -15, 2  # 主控
        ]
    elif preset_name == "延迟重强效果":
        return [
            0.2, 0.4, 0.6, 0.2, 0.3,  # 左声道
            0.2, 0.4, 0.6, 0.3, 0.7, 0.5, 0.2, 0.3,  # 右声道
            -25, 5  # 主控
        ]
    else:
        return [
            0.1, 0.2, 0.7, 0.05, 0.2,  # 左声道
            0.1, 0.2, 0.7, 0.05, 0.1, 0.15, 0.05, 0.2,  # 右声道
            -20, 4  # 主控
        ]


def save_processed_audio(processed_audio_input, save_dir="outputs/processed"):
    """保存处理后的音频文件到指定目录"""
    if processed_audio_input is None:
        return "没有处理好的音频可供保存"
    
    try:
        # 打印输入类型，用于调试
        print(f"保存函数接收到的输入类型: {type(processed_audio_input)}")
        print(f"输入值: {processed_audio_input}")
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(save_dir, f"processed_audio_{timestamp}.wav")
        
        # 处理Gradio音频组件返回的(sample_rate, audio_data)元组
        if isinstance(processed_audio_input, tuple) and len(processed_audio_input) == 2:
            sample_rate, audio_data = processed_audio_input
            
            # 检查是否有效的采样率和音频数据
            if isinstance(sample_rate, (int, float)) and isinstance(audio_data, np.ndarray):
                print(f"使用采样率 {sample_rate} 直接保存音频数据")
                sf.write(output_file, audio_data, int(sample_rate))
                return f"音频已保存至: {output_file}"
            else:
                return f"无效的音频数据格式: ({type(sample_rate)}, {type(audio_data)})"
                
        # 如果是字符串，假设是文件路径
        elif isinstance(processed_audio_input, str):
            if os.path.exists(processed_audio_input):
                # 复制临时文件到目标位置
                import shutil
                shutil.copy2(processed_audio_input, output_file)
                return f"音频文件已复制并保存至: {output_file}"
            else:
                return f"错误: 文件 {processed_audio_input} 不存在"
        else:
            return f"不支持的输入类型: {type(processed_audio_input)}"
        
    except Exception as e:
        import traceback
        return f"保存音频时发生错误: {e}\n{traceback.format_exc()}"


def save_effect_config(
    config_name,
    # 左声道参数
    left_chorus_rate, left_chorus_depth, left_chorus_mix,
    left_reverb_room_size, left_reverb_wet_level,
    # 右声道参数
    right_phaser_rate, right_phaser_depth, right_phaser_mix,
    right_delay_seconds, right_delay_feedback, right_delay_mix,
    right_reverb_room_size, right_reverb_wet_level,
    # 主控参数
    master_threshold_db, master_ratio,
    # 配置保存目录
    config_dir="configs"
):
    """保存当前效果器配置到JSON文件"""
    if not config_name:
        return "请输入配置名称"
    
    try:
        # 创建配置目录
        os.makedirs(config_dir, exist_ok=True)
        
        # 构建配置数据
        config = {
            "name": config_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {
                "left_channel": {
                    "chorus": {
                        "rate": float(left_chorus_rate),
                        "depth": float(left_chorus_depth),
                        "mix": float(left_chorus_mix)
                    },
                    "reverb": {
                        "room_size": float(left_reverb_room_size),
                        "wet_level": float(left_reverb_wet_level)
                    }
                },
                "right_channel": {
                    "phaser": {
                        "rate": float(right_phaser_rate),
                        "depth": float(right_phaser_depth),
                        "mix": float(right_phaser_mix)
                    },
                    "delay": {
                        "seconds": float(right_delay_seconds),
                        "feedback": float(right_delay_feedback),
                        "mix": float(right_delay_mix)
                    },
                    "reverb": {
                        "room_size": float(right_reverb_room_size),
                        "wet_level": float(right_reverb_wet_level)
                    }
                },
                "master": {
                    "compressor": {
                        "threshold_db": float(master_threshold_db),
                        "ratio": float(master_ratio)
                    }
                }
            }
        }
        
        # 生成文件名
        safe_name = "".join(c for c in config_name if c.isalnum() or c in "._- ").rstrip()
        if not safe_name:
            safe_name = "config"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        filepath = os.path.join(config_dir, filename)
        
        # 保存到JSON文件
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return f"配置已保存至: {filepath}"
    
    except Exception as e:
        import traceback
        return f"保存配置时发生错误: {e}\n{traceback.format_exc()}"


def create_gradio_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="音频效果器调参工具") as demo:
        gr.Markdown("# 音频效果器调参工具")
        gr.Markdown("上传音频文件并调整各种效果器参数，实时预览处理后的效果。")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 音频输入区
                gr.Markdown("## 输入音频")
                # 注意：移除了source参数
                audio_input = gr.Audio(label="上传音频文件", type="numpy")
                
                # 预设选择
                preset_dropdown = gr.Dropdown(
                    choices=["默认效果", "强混响效果", "轻微效果", "延迟重强效果"],
                    value="默认效果",
                    label="选择预设效果"
                )
                
                # 保存音频按钮
                save_button = gr.Button("保存处理后的音频")
                save_output = gr.Textbox(label="保存结果")
                
                # 保存配置区域
                gr.Markdown("## 保存效果配置")
                config_name = gr.Textbox(label="配置名称", placeholder="输入一个配置名称...")
                save_config_button = gr.Button("保存当前效果配置")
                save_config_output = gr.Textbox(label="配置保存结果")
                
            with gr.Column(scale=2):
                # 输出区
                gr.Markdown("## 处理后的音频")
                processed_audio = gr.Audio(label="处理后的音频")
                status_output = gr.Textbox(label="处理状态", value="等待处理音频...", interactive=False)
                
                with gr.Accordion("左声道效果器参数", open=False):
                    # 左声道参数控制
                    gr.Markdown("### 左声道 Chorus 参数")
                    left_chorus_rate = gr.Slider(0.01, 2.0, value=0.1, step=0.01, label="Chorus 速率 (Hz)")
                    left_chorus_depth = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Chorus 深度")
                    left_chorus_mix = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Chorus 混合比")
                    
                    gr.Markdown("### 左声道 Reverb 参数")
                    left_reverb_room_size = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Reverb 房间大小")
                    left_reverb_wet_level = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Reverb 湿声电平")
                
                with gr.Accordion("右声道效果器参数", open=False):
                    # 右声道参数控制
                    gr.Markdown("### 右声道 Phaser 参数")
                    right_phaser_rate = gr.Slider(0.01, 2.0, value=0.1, step=0.01, label="Phaser 速率 (Hz)")
                    right_phaser_depth = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Phaser 深度")
                    right_phaser_mix = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Phaser 混合比")
                    
                    gr.Markdown("### 右声道 Delay 参数")
                    right_delay_seconds = gr.Slider(0.01, 1.0, value=0.05, step=0.01, label="Delay 延迟时间 (秒)")
                    right_delay_feedback = gr.Slider(0.0, 0.99, value=0.1, step=0.01, label="Delay 反馈")
                    right_delay_mix = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Delay 混合比")
                    
                    gr.Markdown("### 右声道 Reverb 参数")
                    right_reverb_room_size = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Reverb 房间大小")
                    right_reverb_wet_level = gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Reverb 湿声电平")
                
                with gr.Accordion("主控压缩参数", open=False):
                    # 主控压缩器参数
                    gr.Markdown("### 主控 Compressor 参数")
                    master_threshold_db = gr.Slider(-60, 0, value=-20, step=1, label="压缩阈值 (dB)")
                    master_ratio = gr.Slider(1, 20, value=4, step=0.5, label="压缩比率")
        
        # 处理函数
        process_btn = gr.Button("应用效果")
        
        # 连接处理函数
        process_btn.click(
            fn=process_audio_for_gradio,
            inputs=[
                audio_input,
                # 左声道参数
                left_chorus_rate, left_chorus_depth, left_chorus_mix,
                left_reverb_room_size, left_reverb_wet_level,
                # 右声道参数
                right_phaser_rate, right_phaser_depth, right_phaser_mix,
                right_delay_seconds, right_delay_feedback, right_delay_mix,
                right_reverb_room_size, right_reverb_wet_level,
                # 主控参数
                master_threshold_db, master_ratio
            ],
            outputs=[processed_audio, status_output]
        )
        
        # 连接预设功能
        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[
                # 左声道参数
                left_chorus_rate, left_chorus_depth, left_chorus_mix,
                left_reverb_room_size, left_reverb_wet_level,
                # 右声道参数
                right_phaser_rate, right_phaser_depth, right_phaser_mix,
                right_delay_seconds, right_delay_feedback, right_delay_mix,
                right_reverb_room_size, right_reverb_wet_level,
                # 主控参数
                master_threshold_db, master_ratio
            ]
        )
        
        # 连接保存音频功能
        save_button.click(
            fn=save_processed_audio,
            inputs=[processed_audio],
            outputs=[save_output]
        )
        
        # 连接保存配置功能
        save_config_button.click(
            fn=save_effect_config,
            inputs=[
                config_name,
                # 左声道参数
                left_chorus_rate, left_chorus_depth, left_chorus_mix,
                left_reverb_room_size, left_reverb_wet_level,
                # 右声道参数
                right_phaser_rate, right_phaser_depth, right_phaser_mix,
                right_delay_seconds, right_delay_feedback, right_delay_mix,
                right_reverb_room_size, right_reverb_wet_level,
                # 主控参数
                master_threshold_db, master_ratio
            ],
            outputs=[save_config_output]
        )
    
    return demo


# 使用示例
if __name__ == "__main__":
    # 常规处理和播放
    # output_file = "outputs/results/qianqiuhao_clone/gen_脚本_101_text_processed.wav"
    # success = mono_to_stereo_pedalboard(audio_path, output_file)
    # if success:
    #     play_audio(output_file)
    
    # 启动Gradio界面
    demo = create_gradio_interface()
    demo.launch()  # share=True允许生成一个公共链接以便分享