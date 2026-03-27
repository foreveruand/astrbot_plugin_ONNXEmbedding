"""
ONNX Embedding Provider for AstrBot
基于ONNX Runtime的轻量级嵌入向量生成插件
"""

import asyncio
import gc
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.provider.register import (
    provider_registry,
    register_provider_adapter,
)
from astrbot.core.star.filter.command import GreedyStr

from .rerank_provider import ONNXRerankProvider, register_ONNXRerankProvider

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_ONNX_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
DEFAULT_HUGGINGFACE_MIRROR = ""  # 留空使用官方源


# ============================================================
# 模型下载工具函数
# ============================================================
def _download_file(url: str, output_path: Path, desc: str = ""):
    """下载文件并显示进度"""

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r{desc}: {percent}%")
            sys.stdout.flush()

    logger.info(f"[ONNXEmbedding] 正在下载 {desc}...")
    urlretrieve(url, output_path, reporthook=progress_hook)
    logger.info(f"[ONNXEmbedding] ✓ {desc} 下载完成: {output_path}")


def _download_model_from_hf(
    model_name: str,
    output_dir: Path,
    hf_mirror: str = "",
) -> Path:
    """
    从 Hugging Face 下载 ONNX 模型

    Args:
        model_name: Hugging Face 模型名称（如 Xenova/all-MiniLM-L6-v2）
        output_dir: 输出目录
        hf_mirror: HuggingFace 镜像地址（留空使用官方源）

    Returns:
        模型目录路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = hf_mirror.rstrip("/") if hf_mirror else "https://huggingface.co"
    base_url = f"{base_url}/{model_name}/resolve/main"

    files_to_download = {
        "onnx/model.onnx": "ONNX 模型文件",
        "tokenizer.json": "Tokenizer",
        "config.json": "配置文件",
        "tokenizer_config.json": "Tokenizer 配置",
        "special_tokens_map.json": "特殊 token 映射",
        "vocab.txt": "词表文件",
    }

    logger.info(f"[ONNXEmbedding] 正在下载模型: {model_name}")
    logger.info(f"[ONNXEmbedding] 输出目录: {output_dir}")

    for filename, desc in files_to_download.items():
        output_path = output_dir / filename.replace("onnx/", "")

        if output_path.exists():
            logger.info(f"[ONNXEmbedding] ✓ {desc} 已存在，跳过: {output_path}")
            continue

        url = f"{base_url}/{filename}"

        try:
            _download_file(url, output_path, desc)
        except Exception as e:
            logger.warning(f"[ONNXEmbedding] ✗ {desc} 下载失败: {e}")
            continue

    logger.info(f"[ONNXEmbedding] ✅ 模型下载完成！模型路径: {output_dir}")
    return output_dir


def _mean_pooling(
    last_hidden_state: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """对token embeddings进行均值池化，考虑attention mask"""
    # 扩展attention mask维度以匹配hidden state
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, last_hidden_state.shape)

    # 计算加权和
    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
    mask_sum = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)

    return sum_embeddings / mask_sum


# 全局插件配置（供 Provider 访问）
_PLUGIN_CONFIG: dict = {}


def get_plugin_config() -> dict:
    return _PLUGIN_CONFIG


# ============================================================
# Embedding Provider
# ============================================================
class ONNXEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

        # -------- 下载相关配置（优先从 Provider 配置读取，否则从插件全局配置读取）--------
        self.hf_mirror = provider_config.get(
            "huggingface_mirror",
            _PLUGIN_CONFIG.get("huggingface_mirror", DEFAULT_HUGGINGFACE_MIRROR),
        )
        self.auto_download = (
            provider_config.get("auto_download", _PLUGIN_CONFIG.get("auto_download", 1))
            == 1
        )

        # -------- 自动卸载配置 --------
        self.auto_unload_timeout = provider_config.get(
            "auto_unload_timeout", _PLUGIN_CONFIG.get("auto_unload_timeout", 0)
        )
        self._last_used_time: float = 0
        self._auto_unload_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # -------- 模型路径处理（Pathlib）--------
        base_path = provider_config.get("ONNXEmbedding_path", DEFAULT_MODEL_NAME)

        data_dir = Path(StarTools.get_data_dir("ONNXEmbedding"))
        base_path = Path(base_path)

        self.model_path = base_path if base_path.is_absolute() else data_dir / base_path
        self.tokenizer_path = provider_config.get(
            "ONNXEmbedding_tokenizer_path",
            str(self.model_path / "tokenizer.json"),
        )

        # -------- 运行状态 --------
        self.session = None
        self.tokenizer = None
        self._model_lock = asyncio.Lock()

        # -------- ONNX Runtime 环境检测（一次）--------
        self._env_available = True
        self._env_error: str | None = None
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            self._env_available = False
            self._env_error = "未安装 onnxruntime，请执行：pip install onnxruntime"

        try:
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError:
            self._env_available = False
            self._env_error = "未安装 tokenizers，请执行：pip install tokenizers"

        logger.info(
            f"[ONNXEmbedding] Provider 初始化完成，"
            f"env_available={self._env_available}, "
            f"model_path={self.model_path}"
        )

    # ====================================================
    # 内部工具
    # ====================================================

    def _ensure_env_available(self):
        if not self._env_available:
            raise RuntimeError(f"[ONNXEmbedding] 环境不可用: {self._env_error}")

    def _load_tokenizer(self) -> Any:
        """加载tokenizer并配置padding和truncation"""
        from tokenizers import Tokenizer

        tokenizer_path = Path(self.tokenizer_path)
        if not tokenizer_path.is_absolute():
            tokenizer_path = (
                Path(StarTools.get_data_dir("ONNXEmbedding")) / tokenizer_path
            )

        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            # 尝试在模型目录中查找
            model_dir = Path(self.model_path)
            if model_dir.is_dir():
                for tokenizer_file in ["tokenizer.json", "tokenizer_config.json"]:
                    candidate = model_dir / tokenizer_file
                    if candidate.exists():
                        tokenizer = Tokenizer.from_file(str(candidate))
                        break
                else:
                    raise FileNotFoundError(f"找不到tokenizer文件: {tokenizer_path}")
            else:
                raise FileNotFoundError(f"找不到tokenizer文件: {tokenizer_path}")

        # 从配置中获取最大序列长度，默认为256
        max_length = self.provider_config.get("ONNXEmbedding_max_length", 256)

        # 配置truncation
        tokenizer.enable_truncation(max_length=max_length)

        # 配置padding - pad到max_length
        # 尝试找到合适的pad_token_id
        pad_id = 0  # 默认使用0作为pad_id
        try:
            # 尝试获取[PAD] token的id
            if tokenizer.token_to_id("[PAD]") is not None:
                pad_id = tokenizer.token_to_id("[PAD]")
            elif tokenizer.token_to_id("<pad>") is not None:
                pad_id = tokenizer.token_to_id("<pad>")
        except Exception:
            pass

        tokenizer.enable_padding(pad_id=pad_id, length=max_length)

        return tokenizer

    def _load_onnx_session(self) -> Any:
        model_path = Path(self.model_path)

        if model_path.is_dir():
            onnx_files = list(model_path.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"在目录 {model_path} 中找不到ONNX模型文件")
            model_path = onnx_files[0]

        if not model_path.exists():
            raise FileNotFoundError(f"找不到ONNX模型文件: {model_path}")

        backend = self.provider_config.get("ONNXEmbedding_backend", "auto")

        if backend in ("openvino", "auto"):
            try:
                import openvino as ov

                core = ov.Core()
                devices = core.available_devices
                preferred_device = "GPU" if "GPU" in devices else "CPU"
                logger.info(
                    f"[ONNXEmbedding] OpenVINO 可用设备: {devices}, 使用: {preferred_device}"
                )
                compiled_model = core.compile_model(str(model_path), preferred_device)
                logger.info(f"[ONNXEmbedding] 使用 OpenVINO 后端加载模型成功")
                return ("openvino", compiled_model)
            except ImportError:
                if backend == "openvino":
                    raise RuntimeError("未安装 openvino，请执行：pip install openvino")
                logger.info("[ONNXEmbedding] OpenVINO 未安装，使用 ONNX Runtime")
            except Exception as e:
                if backend == "openvino":
                    raise RuntimeError(f"OpenVINO 加载模型失败: {e}")
                logger.warning(
                    f"[ONNXEmbedding] OpenVINO 加载失败: {e}，回退到 ONNX Runtime"
                )

        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        optimization_level = self.provider_config.get(
            "ONNXEmbedding_optimization_level", "disable"
        )

        if optimization_level == "all":
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
        elif optimization_level == "basic":
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        elif optimization_level == "extended":
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        else:
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )

        providers = ort.get_available_providers()
        preferred_order = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected_providers = [p for p in preferred_order if p in providers]

        logger.info(f"[ONNXEmbedding] ONNX Runtime 提供程序: {selected_providers}")

        disabled_optimizers = None
        if optimization_level == "disable":
            disabled_optimizers = [
                "SimplifiedLayerNormFusion",
                "LayerNormFusion",
                "AttentionFusion",
                "BiasSoftmaxFusion",
            ]

        try:
            return (
                "onnxruntime",
                ort.InferenceSession(
                    str(model_path),
                    sess_options,
                    providers=selected_providers,
                    disabled_optimizers=disabled_optimizers,
                ),
            )
        except Exception as e:
            logger.warning(f"[ONNXEmbedding] 加载模型失败，尝试不优化加载: {e}")
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            return (
                "onnxruntime",
                ort.InferenceSession(
                    str(model_path),
                    sess_options,
                    providers=selected_providers,
                    disabled_optimizers=[
                        "SimplifiedLayerNormFusion",
                        "LayerNormFusion",
                    ],
                ),
            )

    async def _ensure_model_loaded(self):
        self._ensure_env_available()

        if self.session is not None and self.tokenizer is not None:
            return

        async with self._model_lock:
            if self.session is not None and self.tokenizer is not None:
                return

            await self._download_model_if_needed()

            logger.info(f"[ONNXEmbedding] 开始加载模型: {self.model_path}")
            loop = asyncio.get_running_loop()

            try:
                self.tokenizer = await loop.run_in_executor(None, self._load_tokenizer)
                self.session = await loop.run_in_executor(None, self._load_onnx_session)
                self._update_last_used_time()
                self._start_auto_unload_task()
                logger.info("[ONNXEmbedding] 模型加载成功")
            except Exception as e:
                logger.error("[ONNXEmbedding] 模型加载失败", exc_info=True)
                raise RuntimeError(f"模型加载失败: {e}") from e

    async def _download_model_if_needed(self):
        if self._check_model_exists():
            return

        if not self.auto_download:
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}，且未启用自动下载。"
                f"请手动下载模型或启用自动下载功能。"
            )

        model_name = self._extract_model_name_from_path()
        if not model_name:
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}，且无法从路径推断模型名称。"
                f"请手动下载模型或使用 HuggingFace 模型名称作为路径。"
            )

        logger.info(f"[ONNXEmbedding] 模型不存在，开始自动下载: {model_name}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _download_model_from_hf,
            model_name,
            self.model_path,
            self.hf_mirror,
        )

    def _check_model_exists(self) -> bool:
        model_path = Path(self.model_path)
        if model_path.is_file():
            return True
        if model_path.is_dir():
            onnx_files = list(model_path.glob("*.onnx"))
            if onnx_files:
                return True
        return False

    def _extract_model_name_from_path(self) -> str | None:
        path_str = str(self.model_path)
        if "/" in path_str or "\\" in path_str:
            parts = path_str.replace("\\", "/").rstrip("/").split("/")
            if len(parts) >= 2:
                return f"{parts[-2]}/{parts[-1]}"
            return parts[-1]
        return path_str

    def _cleanup_resources(self) -> bool:
        try:
            self.session = None
            self.tokenizer = None
            gc.collect()
            return True
        except Exception:
            logger.error("[ONNXEmbedding] 资源清理失败", exc_info=True)
            return False

    def _update_last_used_time(self):
        self._last_used_time = time.time()

    def _start_auto_unload_task(self):
        logger.info(
            f"[ONNXEmbedding] 自动卸载超时配置: {self.auto_unload_timeout} 分钟"
        )
        if self.auto_unload_timeout <= 0:
            logger.info("[ONNXEmbedding] 自动卸载未启用（auto_unload_timeout <= 0）")
            return
        if self._auto_unload_task is not None and not self._auto_unload_task.done():
            return

        self._shutdown_event.clear()
        self._auto_unload_task = asyncio.create_task(self._auto_unload_loop())
        logger.info(
            f"[ONNXEmbedding] 已启动自动卸载任务，超时时间: {self.auto_unload_timeout} 分钟"
        )

    async def _auto_unload_loop(self):
        check_interval = min(60, self.auto_unload_timeout * 60 / 2)
        check_interval = max(10, check_interval)

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(check_interval)

                if self.session is None:
                    continue

                elapsed = time.time() - self._last_used_time
                timeout_seconds = self.auto_unload_timeout * 60

                if elapsed >= timeout_seconds:
                    logger.info(
                        f"[ONNXEmbedding] 模型已 {self.auto_unload_timeout} 分钟未使用，自动卸载"
                    )
                    async with self._model_lock:
                        if self.session is not None:
                            self._cleanup_resources()
                            logger.info("[ONNXEmbedding] 模型已自动卸载")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ONNXEmbedding] 自动卸载检查出错: {e}")

    def _stop_auto_unload_task(self):
        if self._auto_unload_task is not None:
            self._shutdown_event.set()
            if not self._auto_unload_task.done():
                self._auto_unload_task.cancel()
            self._auto_unload_task = None

    # ====================================================
    # Embedding API
    # ====================================================

    async def get_embedding(self, text: str) -> list[float]:
        await self._ensure_model_loaded()
        self._update_last_used_time()

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self._encode, [text])
        return embedding[0].tolist()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        await self._ensure_model_loaded()
        self._update_last_used_time()

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self._encode, texts)
        return embeddings.tolist()

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.tokenizer is None or self.session is None:
            raise RuntimeError("模型未加载")

        encoding = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoding], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoding], dtype=np.int64)

        backend, model = self.session

        if backend == "openvino":
            return self._encode_openvino(model, input_ids, attention_mask)
        else:
            return self._encode_onnxruntime(model, input_ids, attention_mask)

    def _encode_openvino(
        self, model, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        import openvino as ov

        input_names = [inp.any_name for inp in model.inputs]
        feed_dict = {}

        if "input_ids" in input_names:
            feed_dict["input_ids"] = ov.Tensor(input_ids)
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = ov.Tensor(attention_mask)
        if "token_type_ids" in input_names:
            feed_dict["token_type_ids"] = ov.Tensor(np.zeros_like(input_ids))

        infer_request = model.create_infer_request()
        infer_request.infer(feed_dict)

        output_names = [out.any_name for out in model.outputs]

        if "sentence_embedding" in output_names:
            embeddings = infer_request.get_tensor("sentence_embedding").data[:]
        elif "last_hidden_state" in output_names:
            last_hidden_state = infer_request.get_tensor("last_hidden_state").data[:]
            embeddings = _mean_pooling(last_hidden_state, attention_mask)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            embeddings = list(infer_request.results.values())[0].data[:]
            if len(embeddings.shape) == 3:
                embeddings = _mean_pooling(embeddings, attention_mask)
                embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )

        return embeddings

    def _encode_onnxruntime(
        self, session, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        input_names = [inp.name for inp in session.get_inputs()]
        feed_dict = {}

        if "input_ids" in input_names:
            feed_dict["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            feed_dict["token_type_ids"] = np.zeros_like(input_ids)

        outputs = session.run(None, feed_dict)
        output_names = [out.name for out in session.get_outputs()]

        if "sentence_embedding" in output_names:
            idx = output_names.index("sentence_embedding")
            embeddings = outputs[idx]
        elif "last_hidden_state" in output_names:
            idx = output_names.index("last_hidden_state")
            last_hidden_state = outputs[idx]
            embeddings = _mean_pooling(last_hidden_state, attention_mask)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            embeddings = outputs[0]
            if len(embeddings.shape) == 3:
                embeddings = _mean_pooling(embeddings, attention_mask)
                embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )

        return embeddings

    def get_dim(self) -> int:
        """获取嵌入向量维度"""
        return int(self.provider_config.get("embedding_dimensions", 384))

    # ====================================================
    # 卸载
    # ====================================================

    async def unload_model(self) -> bool:
        self._stop_auto_unload_task()
        async with self._model_lock:
            if self.session is None:
                logger.info("[ONNXEmbedding] 模型未加载，无需卸载")
                return True

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._cleanup_resources)

    def force_unload_sync(self) -> bool:
        self._stop_auto_unload_task()
        if self.session is None:
            return True
        return self._cleanup_resources()


# ============================================================
# Provider 注册函数（只负责注册）
# ============================================================
def register_ONNXEmbeddingProvider():
    try:
        register_provider_adapter(
            "ONNXEmbedding",
            "ONNX Runtime Embedding Provider",
            provider_type=ProviderType.EMBEDDING,
        )(ONNXEmbeddingProvider)
        logger.info("[ONNXEmbedding] Provider 已注册")
    except ValueError:
        logger.info("[ONNXEmbedding] Provider 已存在，跳过注册")


# ============================================================
# Star 插件本体
# ============================================================
@register("ONNXEmbedding", "AstrBot", "ONNX Embedding Provider", "1.0.0")
class ONNXEmbedding(Star):
    _registered = False

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.auto_start = self.config.get("auto_start", 0) == 1

        global _PLUGIN_CONFIG
        _PLUGIN_CONFIG = dict(self.config)

    # --------------------------------------------------------
    def _register_config(self):
        if self._registered:
            return False
        # ---- 防御性获取配置节点----
        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                "config_template"
            ]["ONNXEmbedding"] = {
                "id": "ONNXEmbedding",
                "type": "ONNXEmbedding",
                "provider": "Local",
                "ONNXEmbedding_path": DEFAULT_MODEL_NAME,
                "huggingface_mirror": self.config.get("huggingface_mirror", ""),
                "auto_download": self.config.get("auto_download", 1),
                "auto_unload_timeout": self.config.get("auto_unload_timeout", 0),
                "provider_type": "embedding",
                "enable": True,
                "embedding_dimensions": 384,
            }
        except KeyError:
            logger.error("[ONNXEmbedding] AstrBot 配置结构异常，无法注册 Provider")
            return False

        # ---- Provider 注册 ----
        already_registered = False
        if isinstance(provider_registry, list):
            for p in provider_registry:
                if getattr(p, "type", None) == "ONNXEmbedding":
                    already_registered = True
                    break

        if not already_registered:
            register_ONNXEmbeddingProvider()

        # ---- Rerank Provider 注册 ----
        rerank_already_registered = False
        if isinstance(provider_registry, list):
            for p in provider_registry:
                if getattr(p, "type", None) == "ONNXRerank":
                    rerank_already_registered = True
                    break

        if not rerank_already_registered:
            register_ONNXRerankProvider()
            try:
                CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                    "config_template"
                ]["ONNXRerank"] = {
                    "id": "ONNXRerank",
                    "type": "ONNXRerank",
                    "provider": "Local",
                    "ONNXRerank_path": "BAAI/bge-reranker-base",
                    "provider_type": "rerank",
                    "enable": True,
                }
            except KeyError:
                logger.warning("[ONNXRerank] 无法注册 Rerank Provider 配置")

        self._registered = True
        logger.info("[ONNXEmbedding] 配置与 Provider 注册完成")
        return True

    def _unregister_config(self):
        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                "config_template"
            ].pop("ONNXEmbedding", None)
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "ONNXEmbedding_path", None
            )
        except KeyError:
            pass

        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                "config_template"
            ].pop("ONNXRerank", None)
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "ONNXRerank_path", None
            )
        except KeyError:
            pass

        self._registered = False
        logger.info("[ONNXEmbedding] 配置已清理")

    # --------------------------------------------------------
    # Commands
    # --------------------------------------------------------
    @filter.command("onnx")
    async def query_kb(
        self, event: AstrMessageEvent, kb_name: str, query_text: GreedyStr = ""
    ):
        if not query_text:
            yield event.plain_result(
                "[ONNXEmbedding] 用法: /onnx <知识库名> <查询内容>"
            )
            return

        kb_manager = self.context.kb_manager
        kb_helper = await kb_manager.get_kb_by_name(kb_name)
        if not kb_helper:
            yield event.plain_result(f"[ONNXEmbedding] 未找到知识库: {kb_name}")
            return

        try:
            results = await kb_manager.retrieve(
                query=query_text,
                kb_names=[kb_name],
                top_k_fusion=10,
                top_m_final=2,
            )

            if not results or not results.get("results"):
                yield event.plain_result(f"[ONNXEmbedding] 未找到相关内容")
                return

            output_lines = [f"查询结果 (知识库: {kb_name}):"]
            for i, r in enumerate(results["results"], 1):
                content = r.get("content", "")
                score = r.get("score", 0)
                doc_name = r.get("doc_name", "未知文档")
                output_lines.append(f"\n【结果 {i}】")
                output_lines.append(f"来源: {doc_name}")
                output_lines.append(f"相关度: {score:.4f}")
                output_lines.append(
                    f"内容: {content[:200]}{'...' if len(content) > 200 else ''}"
                )

            yield event.plain_result("\n".join(output_lines))

        except Exception as e:
            logger.error(f"[ONNXEmbedding] 查询失败: {e}", exc_info=True)
            yield event.plain_result(f"[ONNXEmbedding] 查询失败: {e}")

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------
    async def initialize(self):
        if not self.auto_start:
            logger.info("[ONNXEmbedding] 未启用自加载")
            return
        logger.info("[ONNXEmbedding] 插件初始化中")
        if self._register_config():
            logger.info("[ONNXEmbedding] 注册 Provider 成功")
        else:
            logger.error("[ONNXEmbedding] 插件初始化失败")

    async def terminate(self):
        logger.info("[ONNXEmbedding] 插件终止中")
        self._unregister_config()
        logger.info("[ONNXEmbedding] 插件终止完成")

    # --------------------------------------------------------
    # 在astrbot启动时
    # --------------------------------------------------------
    @filter.on_astrbot_loaded()
    async def init_db(self):
        """如果启动自动加载，将在astrbot启动后自动刷新数据库"""
        if not self.auto_start:
            return
        if not self._registered:
            logger.info("[ONNXEmbedding] 刷新数据库失败,未注册编码器")
        try:
            await self.context.kb_manager.load_kbs()
            logger.info("[ONNXEmbedding] 插件初始化完成,已重新刷新数据库")
        except Exception:
            raise
