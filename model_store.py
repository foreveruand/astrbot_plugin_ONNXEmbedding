"""Unified model download, existence check, and manifest utilities.

All three provider types (embedding, rerank, chat) share this module so that
download logic, progress display, HuggingFace mirror handling, and manifest
writing are implemented exactly once.
"""

import json
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

from astrbot.api import logger

# ---------------------------------------------------------------------------
# File manifests for each model category
# Each entry: (hf_relative_path, description, required)
# ---------------------------------------------------------------------------

#: Files expected for ONNX embedding models (Sentence-Transformers style)
EMBEDDING_FILES: list[tuple[str, str, bool]] = [
    ("onnx/model.onnx", "ONNX 模型文件", True),
    ("tokenizer.json", "Tokenizer", True),
    ("config.json", "配置文件", True),
    ("tokenizer_config.json", "Tokenizer 配置", False),
    ("special_tokens_map.json", "特殊 token 映射", False),
    ("vocab.txt", "词表文件", False),
]

#: Files expected for ONNX rerank models (Cross-Encoder style)
RERANK_FILES: list[tuple[str, str, bool]] = [
    ("onnx/model.onnx", "ONNX Rerank 模型文件", True),
    ("tokenizer.json", "Tokenizer", True),
    ("config.json", "配置文件", True),
    ("tokenizer_config.json", "Tokenizer 配置", False),
    ("special_tokens_map.json", "特殊 token 映射", False),
]

#: Files downloaded for GenAI models when huggingface_hub is unavailable (fallback only).
#: Prefer ``download_chat_model()`` which downloads model weights via snapshot_download.
CHAT_FILES: list[tuple[str, str, bool]] = [
    ("config.json", "模型配置", True),
    ("tokenizer.json", "Tokenizer", True),
    ("tokenizer_config.json", "Tokenizer 配置", False),
    ("special_tokens_map.json", "特殊 token 映射", False),
    ("generation_config.json", "生成配置", False),
    # OpenVINO IR weights (present on Intel HF repos)
    ("openvino_model.xml", "OpenVINO 模型描述 (xml)", False),
    ("openvino_model.bin", "OpenVINO 模型权重 (bin)", False),
    # ORT GenAI weights
    ("model.onnx", "ONNX 模型文件", False),
    ("model.onnx.data", "ONNX 模型数据 (large)", False),
]

_MANIFEST_FILENAME = ".onnx_manifest.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, output_path: Path, desc: str = "") -> None:
    """Download *url* to *output_path*, showing a compact progress bar."""

    def _hook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, int(count * block_size * 100 / total_size))
            sys.stdout.write(f"\r  [{desc}] {pct}%  ")
            sys.stdout.flush()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(output_path), reporthook=_hook)
    sys.stdout.write("\n")
    logger.info(f"[ModelStore] ✓ {desc} 下载完成: {output_path}")


def _write_manifest(model_dir: Path, model_name: str) -> None:
    manifest = {
        "model_name": model_name,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        (model_dir / _MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"[ModelStore] 无法写入 manifest: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_model(
    model_name: str,
    output_dir: Path,
    file_list: list[tuple[str, str, bool]],
    hf_mirror: str = "",
) -> tuple[bool, list[str]]:
    """Download model files from HuggingFace into *output_dir*.

    Files that already exist are skipped (idempotent).  Required files that
    fail to download cause the function to return ``(False, [failed_paths])``.
    Optional file failures are logged and ignored.

    Args:
        model_name: HuggingFace model slug, e.g. ``"BAAI/bge-reranker-base"``.
        output_dir: Local directory to write files into.
        file_list: List of ``(hf_path, description, required)`` triples.
        hf_mirror: Optional mirror base URL (default uses huggingface.co).

    Returns:
        ``(success, failed_required_paths)``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = hf_mirror.rstrip("/") if hf_mirror else "https://huggingface.co"
    base_url = f"{base}/{model_name}/resolve/main"

    failed_required: list[str] = []
    logger.info(f"[ModelStore] 开始下载模型: {model_name} → {output_dir}")

    for hf_path, desc, required in file_list:
        # Flatten onnx/ prefix: store model.onnx at the root of output_dir
        local_name = hf_path.replace("onnx/", "")
        dest = output_dir / local_name

        if dest.exists():
            logger.debug(f"[ModelStore] ✓ {desc} 已存在，跳过")
            continue

        url = f"{base_url}/{hf_path}"
        try:
            _download_file(url, dest, desc)
        except Exception as exc:
            msg = f"{desc} ({hf_path}): {exc}"
            if required:
                logger.error(f"[ModelStore] ✗ 必需文件下载失败: {msg}")
                failed_required.append(hf_path)
            else:
                logger.warning(f"[ModelStore] ✗ 可选文件下载失败（忽略）: {msg}")

    if failed_required:
        logger.error(
            f"[ModelStore] 模型 {model_name} 下载不完整，"
            f"缺少必需文件: {failed_required}"
        )
        return False, failed_required

    _write_manifest(output_dir, model_name)
    logger.info(f"[ModelStore] ✅ 模型 {model_name} 下载完成: {output_dir}")
    return True, []


def download_chat_model(
    model_name: str,
    output_dir: Path,
    hf_mirror: str = "",
    backend: str = "auto",
    force_download: bool = False,
) -> tuple[bool, list[str]]:
    """Download a local chat model into *output_dir* using Hugging Face Hub.

    This now follows the Hugging Face OpenVINO docs more closely by preferring
    ``snapshot_download(local_dir=...)`` for complete repo checkout. That keeps
    OpenVINO bundles such as ``openvino_model.xml/.bin`` and tokenizer files in
    the exact structure expected by ``openvino_genai.LLMPipeline``.

    Args:
        model_name: HuggingFace model slug.
        output_dir: Local directory to write files into.
        hf_mirror: Optional HuggingFace endpoint/mirror.
        backend: Kept for API compatibility.
        force_download: Refresh files even if they already exist locally.

    Returns:
        ``(success, error_messages)``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ignore_patterns = [
        "original/*",
        "pytorch_model*.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.msgpack",
        "*.h5",
        "flax_model*",
        "tf_model*.h5",
        "rust_model.ot",
        "training_args.bin",
    ]

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download

        endpoint = hf_mirror.rstrip("/") if hf_mirror else None
        snapshot_kwargs: dict[str, object] = {
            "repo_id": model_name,
            "local_dir": str(output_dir),
            "ignore_patterns": ignore_patterns,
            "force_download": force_download,
        }
        if endpoint:
            snapshot_kwargs["endpoint"] = endpoint

        logger.info(
            f"[ModelStore] 开始下载 Chat 模型 (snapshot_download): {model_name} → {output_dir}"
        )
        try:
            # Newer huggingface_hub versions support local_dir_use_symlinks=False;
            # keep a compatibility fallback for older releases.
            snapshot_download(local_dir_use_symlinks=False, **snapshot_kwargs)
        except TypeError:
            snapshot_download(**snapshot_kwargs)

        if not check_model_exists(output_dir):
            # Fallback: explicitly download the listed repo files into local_dir.
            api = HfApi(endpoint=endpoint) if endpoint else HfApi()
            repo_files = api.list_repo_files(model_name, repo_type="model")
            logger.warning(
                f"[ModelStore] snapshot_download 后未发现模型权重，尝试逐文件补全 ({len(repo_files)} 个文件)"
            )
            for repo_file in repo_files:
                lower = repo_file.lower()
                basename = Path(repo_file).name.lower()
                if repo_file.startswith("original/"):
                    continue
                if basename.startswith(("pytorch_model", "flax_model", "tf_model")):
                    continue
                if basename in {
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "training_args.bin",
                    "rust_model.ot",
                }:
                    continue
                if lower.endswith((".msgpack", ".h5", ".safetensors")):
                    continue

                hf_hub_download(
                    repo_id=model_name,
                    filename=repo_file,
                    repo_type="model",
                    local_dir=str(output_dir),
                    endpoint=endpoint,
                    force_download=force_download,
                )

        _write_manifest(output_dir, model_name)
        if not check_model_exists(output_dir):
            logger.error(f"[ModelStore] 下载完成但未发现模型权重文件: {output_dir}")
            return False, ["下载完成但未发现 .onnx / .xml 模型文件"]

        logger.info(f"[ModelStore] ✅ Chat 模型下载完成: {output_dir}")
        return True, []

    except ImportError:
        logger.warning(
            "[ModelStore] huggingface_hub 未安装，回退到逐文件下载。"
            "建议: pip install 'huggingface_hub>=0.20'"
        )
        return download_model(model_name, output_dir, CHAT_FILES, hf_mirror)

    except Exception as exc:
        logger.error(f"[ModelStore] Chat 模型下载失败: {exc}", exc_info=True)
        return False, [str(exc)]


def check_model_exists(model_dir: Path) -> bool:
    """Return True if *model_dir* contains a usable ONNX/OpenVINO model bundle."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return False

    # Any ONNX file counts as present. Some repos also include split `.onnx_data*`
    # shards, but the base `.onnx` file is the key signal.
    if any(True for _ in model_dir.rglob("*.onnx")):
        return True

    # OpenVINO models require a matching `.xml` + `.bin` pair. Reject tiny bin
    # files as they are usually pointer/partial downloads and will fail at load time.
    for xml_file in model_dir.rglob("*.xml"):
        bin_file = xml_file.with_suffix(".bin")
        if bin_file.exists() and bin_file.stat().st_size > 4096:
            return True

    return False


def read_manifest(model_dir: Path) -> dict | None:
    """Read and return the manifest dict from *model_dir*, or None."""
    path = Path(model_dir) / _MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_model_name(model_path: str | Path) -> str | None:
    """Attempt to extract a HuggingFace ``org/model`` slug from a path.

    Returns the last two path components joined by ``/``, or just the last
    component if the path is a single segment.  May return ``None`` only when
    the path is empty.
    """
    path_str = str(model_path).replace("\\", "/").rstrip("/")
    if not path_str:
        return None
    parts = path_str.split("/")
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-2]}/{parts[-1]}"


def detect_model_precision(model_dir: Path) -> str:
    """Return the quantization precision of the model in *model_dir*.

    Possible return values: ``"INT8"``, ``"FP16"``, ``"FP32"``, ``"unknown"``.
    Requires the ``onnx`` package for ONNX inspection; falls back gracefully.
    """
    model_dir = Path(model_dir)
    onnx_file: Path | None = None
    if model_dir.is_file() and model_dir.suffix == ".onnx":
        onnx_file = model_dir
    elif model_dir.is_dir():
        onnx_files = list(model_dir.glob("*.onnx"))
        if onnx_files:
            onnx_file = onnx_files[0]

    if onnx_file is None:
        return "unknown"

    try:
        import onnx

        model = onnx.load(str(onnx_file))
        op_types = {node.op_type for node in model.graph.node}
        # Common quantization op names
        if any(
            kw in op for op in op_types for kw in ("QLinear", "Quant", "QDQ", "Dequant")
        ):
            return "INT8"
        # Check initializer element types: FLOAT16=10
        for init in model.graph.initializer:
            if init.data_type == 10:
                return "FP16"
        return "FP32"
    except ImportError:
        return "unknown"
    except Exception as exc:
        logger.debug(f"[ModelStore] 无法检测模型精度: {exc}")
        return "unknown"
