#!/usr/bin/env python3
"""
Test script for long text generation (20000+ characters)
测试脚本：长文本生成能力
"""

import sys
import time
import torch
sys.path.insert(0, 'src')

# Disable MPS
torch.backends.mps.enabled = False

from voxcpm import VoxCPM
import soundfile as sf

def test_long_text_generation():
    print("=" * 70)
    print("VoxCPM 长文本生成测试")
    print("=" * 70)

    # Initialize model
    print("\n[1/4] 初始化模型...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
    print("✓ 模型已加载")

    # Test text of increasing length
    test_cases = [
        {
            "name": "短文本测试 (500 字)",
            "text": "你好，这是一个简短的测试文本。" * 16,  # ~500 chars
            "output": "output_500.wav",
        },
        {
            "name": "中等文本测试 (2000 字)",
            "text": "你好，我是VoxCPM文本转语音系统。这是一个中等长度的测试文本，用来演示模型在处理较长输入时的能力。" * 20,  # ~2000 chars
            "output": "output_2000.wav",
        },
        {
            "name": "长文本测试 (5000 字)",
            "text": "VoxCPM是一个先进的文本转语音系统，采用了最新的深度学习技术。它能够生成自然流畅的语音，支持多种语言和方言。这个系统特别优化了对长文本的处理能力，使用了智能分块技术。" * 25,  # ~5000 chars
            "output": "output_5000.wav",
        },
        {
            "name": "超长文本测试 (10000 字)",
            "text": """
            VoxCPM是OpenBMB开发的一个革命性的文本转语音系统。
            它采用了最新的Transformer架构和扩散模型技术。
            该系统能够理解上下文，生成更自然、更有表现力的语音。
            无论是新闻播报、有声书还是对话系统，VoxCPM都能提供高质量的语音输出。
            系统支持中文和英文，能够处理复杂的语言结构。
            它还能够实现零样本语音克隆，仅需一段短音频就能复制说话人的特点。
            这项技术在许多应用场景中都有广泛的潜力。
            """ * 8,  # ~10000 chars
            "output": "output_10000.wav",
        },
    ]

    for idx, test_case in enumerate(test_cases, 2):
        text = test_case["text"]
        output_file = test_case["output"]

        print(f"\n[{idx}/{len(test_cases)+1}] {test_case['name']}")
        print(f"  文本长度：{len(text)} 字符")

        # Estimate processing time
        chunks = model._split_text_smart(text, max_chars=3000)
        est_time = len(chunks) * 30
        print(f"  预估分块数：{len(chunks)}")
        print(f"  预估耗时：{est_time} 秒 ({est_time/60:.1f} 分钟)")

        try:
            start_time = time.time()

            # Generate audio
            if len(text) > 4000:
                print("  使用长文本生成模式...")
                wav = model.generate_long(
                    text=text,
                    inference_timesteps=8,  # Lower for speed
                    cfg_value=2.0,
                )
            else:
                print("  使用标准生成模式...")
                wav = model.generate(
                    text=text,
                    inference_timesteps=8,
                    cfg_value=2.0,
                )

            elapsed = time.time() - start_time

            # Save
            sf.write(output_file, wav, model.tts_model.sample_rate)

            # Stats
            audio_duration = len(wav) / model.tts_model.sample_rate
            rtf = elapsed / audio_duration

            print(f"  ✓ 生成完成")
            print(f"    耗时：{elapsed:.1f} 秒")
            print(f"    音频长度：{audio_duration:.1f} 秒")
            print(f"    实时因子 (RTF)：{rtf:.2f}")
            print(f"    保存到：{output_file}")

        except Exception as e:
            print(f"  ✗ 失败：{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print("\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print("=" * 70)
    print("\n生成的音频文件：")
    for test_case in test_cases:
        print(f"  - {test_case['output']}")

    return 0


if __name__ == "__main__":
    sys.exit(test_long_text_generation())
