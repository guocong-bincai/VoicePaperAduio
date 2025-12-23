#!/usr/bin/env python3
"""
VoxCPM 文本分割工具 - 处理超长文本
将大于 4096 Token 的文本自动分割成可处理的块
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def split_by_sentences(text: str, max_chars: int = 2000, language: str = "auto"):
    """
    按句子分割文本

    Args:
        text: 输入文本
        max_chars: 每个块的最大字符数
        language: 'zh' (中文), 'en' (英文), 'auto' (自动检测)

    Returns:
        list: 分割后的文本块列表
    """
    if language == "auto":
        # 简单的自动检测
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            language = "zh"
        else:
            language = "en"

    chunks = []
    current_chunk = ""

    if language == "zh":
        # 中文：按句号、问号、感叹号分割
        sentences = []
        temp = ""
        for char in text:
            temp += char
            if char in '。！？\n':
                sentences.append(temp)
                temp = ""
        if temp:
            sentences.append(temp)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

    else:  # 英文或其他
        # 英文：按句号、问号、感叹号分割
        sentences = []
        temp = ""
        for char in text:
            temp += char
            if char in '.!?\n':
                sentences.append(temp)
                temp = ""
        if temp:
            sentences.append(temp)

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def estimate_tokens(text: str, language: str = "auto"):
    """
    估算文本需要的 Token 数

    Args:
        text: 输入文本
        language: 语言

    Returns:
        int: 估算的 Token 数
    """
    if language == "auto":
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            language = "zh"
        else:
            language = "en"

    if language == "zh":
        # 中文：大约 1 字符 ≈ 1 Token
        return len(text)
    else:
        # 英文：大约 1 字符 ≈ 0.25 Token (粗略估算)
        return int(len(text) * 0.25)


def format_output(chunks: list):
    """格式化输出分割结果"""
    print("\n" + "=" * 60)
    print(f"📊 文本分割结果")
    print("=" * 60)
    print(f"总块数：{len(chunks)}")
    print(f"字符数：{sum(len(c) for c in chunks)}")

    for i, chunk in enumerate(chunks, 1):
        tokens = estimate_tokens(chunk)
        print(f"\n【块 {i}】")
        print(f"  字符数：{len(chunk)}")
        print(f"  估算 Token：{tokens}")
        print(f"  预览：{chunk[:100]}{'...' if len(chunk) > 100 else ''}")

    print("\n" + "=" * 60)
    print("✅ 分割完成")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="VoxCPM 文本分割工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从命令行输入
  python text_splitter.py --text "很长的中文文本..."

  # 从文件读取
  python text_splitter.py --file input.txt

  # 指定最大长度
  python text_splitter.py --file input.txt --max-chars 1500

  # 自动检测语言
  python text_splitter.py --text "混合 text 中文"
        """
    )

    parser.add_argument("--text", type=str, help="输入文本（直接）")
    parser.add_argument("--file", type=str, help="输入文本文件路径")
    parser.add_argument("--max-chars", type=int, default=2000, help="每块最大字符数（默认 2000）")
    parser.add_argument("--language", choices=["zh", "en", "auto"], default="auto", help="语言（默认自动检测）")
    parser.add_argument("--save", type=str, help="保存分割结果到文件（JSON 格式）")

    args = parser.parse_args()

    # 获取输入文本
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"❌ 文件未找到：{args.file}")
            return 1
    else:
        print("❌ 请使用 --text 或 --file 指定输入")
        parser.print_help()
        return 1

    # 分割文本
    chunks = split_by_sentences(text, max_chars=args.max_chars, language=args.language)

    # 输出结果
    format_output(chunks)

    # 保存结果
    if args.save:
        import json
        result = {
            "total_chunks": len(chunks),
            "total_chars": sum(len(c) for c in chunks),
            "estimated_tokens": sum(estimate_tokens(c, args.language) for c in chunks),
            "chunks": [
                {
                    "index": i,
                    "text": chunk,
                    "chars": len(chunk),
                    "tokens": estimate_tokens(chunk, args.language)
                }
                for i, chunk in enumerate(chunks, 1)
            ]
        }
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到：{args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
