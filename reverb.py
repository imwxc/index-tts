import soundfile as sf
from pedalboard import Pedalboard, Reverb

key = 'output'

# 加载音频文件
audio, sample_rate = sf.read(f'outputs/{key}.wav')

# 创建效果板并添加混响
board = Pedalboard([
    Reverb(
        room_size=0.1,     # 房间大小，0.0到1.0
        damping=0.5,       # 阻尼
        wet_level=0.33,    # 混响信号比例
        dry_level=0.4      # 原始信号比例
    )
])

# 应用效果
effected = board(audio, sample_rate)

# 保存结果
sf.write(f'{key}_reverb.wav', effected, sample_rate)
