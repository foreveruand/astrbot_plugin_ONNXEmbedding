"""下载 Hugging Face ONNX 模型的脚本"""

import sys
from pathlib import Path
from urllib.request import urlretrieve


def download_file(url: str, output_path: Path, desc: str = ""):
    """下载文件并显示进度"""

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r{desc}: {percent}%")
        sys.stdout.flush()

    print(f"Downloading {desc}...")
    urlretrieve(url, output_path, reporthook=progress_hook)
    print(f"\n✓ {desc} 下载完成: {output_path}")


def download_model(
    model_name: str = "Xenova/all-MiniLM-L6-v2", output_dir: Path = None
):
    """
    从 Hugging Face 下载 ONNX 模型

    Args:
        model_name: Hugging Face 模型名称
        output_dir: 输出目录，默认为 ./models/{model_name}
    """
    if output_dir is None:
        output_dir = Path("models") / model_name.replace("/", "_")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/{model_name}/resolve/main"

    # 需要下载的文件
    files_to_download = {
        "onnx/model.onnx": "ONNX 模型文件",
        "tokenizer.json": "Tokenizer",
        "config.json": "配置文件",
        "tokenizer_config.json": "Tokenizer 配置",
        "special_tokens_map.json": "特殊 token 映射",
        "vocab.txt": "词表文件",
    }

    print(f"\n正在下载模型: {model_name}")
    print(f"输出目录: {output_dir.absolute()}\n")

    for filename, desc in files_to_download.items():
        output_path = output_dir / filename.replace("onnx/", "")

        # 如果文件已存在，跳过
        if output_path.exists():
            print(f"✓ {desc} 已存在，跳过: {output_path}")
            continue

        url = f"{base_url}/{filename}"

        try:
            download_file(url, output_path, desc)
        except Exception as e:
            print(f"✗ {desc} 下载失败: {e}")
            # 某些文件可能不存在，继续下载其他文件
            continue

    print("\n✅ 模型下载完成！")
    print(f"模型路径: {output_dir}")
    print("\n在 AstrBot 配置中使用以下路径:")
    print(f"  ONNXEmbedding_path: {output_dir.absolute()}")
    print(
        f"  ONNXEmbedding_tokenizer_path: {(output_dir / 'tokenizer.json').absolute()}"
    )

    return output_dir


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="下载 Hugging Face ONNX 模型")
    parser.add_argument(
        "--model",
        type=str,
        default="Xenova/all-MiniLM-L6-v2",
        help="Hugging Face 模型名称 (默认: Xenova/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录 (默认: ./models/{model_name})",
    )

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None
    download_model(args.model, output_dir)


if __name__ == "__main__":
    main()
