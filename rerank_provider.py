"""
ONNX Rerank Provider for AstrBot
基于ONNX Runtime的重排序模型Provider
"""

import asyncio
import gc
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from astrbot.api import logger
from astrbot.core.provider.entities import ProviderType, RerankResult
from astrbot.core.provider.provider import RerankProvider
from astrbot.core.provider.register import provider_cls_map, register_provider_adapter
from astrbot.core.star import StarTools

from ._plugin_config import get_plugin_config

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


def _download_rerank_file(url: str, output_path: Path, desc: str = ""):
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r{desc}: {percent}%")
            sys.stdout.flush()

    logger.info(f"[ONNXRerank] 正在下载 {desc}...")
    urlretrieve(url, output_path, reporthook=progress_hook)
    logger.info(f"[ONNXRerank] ✓ {desc} 下载完成: {output_path}")


def _download_rerank_model_from_hf(
    model_name: str,
    output_dir: Path,
    hf_mirror: str = "",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = hf_mirror.rstrip("/") if hf_mirror else "https://huggingface.co"
    base_url = f"{base_url}/{model_name}/resolve/main"

    files_to_download = {
        "onnx/model.onnx": "ONNX Rerank 模型文件",
        "tokenizer.json": "Tokenizer",
        "config.json": "配置文件",
        "tokenizer_config.json": "Tokenizer 配置",
        "special_tokens_map.json": "特殊 token 映射",
    }

    logger.info(f"[ONNXRerank] 正在下载 Rerank 模型: {model_name}")
    logger.info(f"[ONNXRerank] 输出目录: {output_dir}")

    for filename, desc in files_to_download.items():
        output_path = output_dir / filename.replace("onnx/", "")

        if output_path.exists():
            logger.info(f"[ONNXRerank] ✓ {desc} 已存在，跳过: {output_path}")
            continue

        url = f"{base_url}/{filename}"

        try:
            _download_rerank_file(url, output_path, desc)
        except Exception as e:
            logger.warning(f"[ONNXRerank] ✗ {desc} 下载失败: {e}")
            continue

    logger.info(f"[ONNXRerank] ✅ Rerank 模型下载完成！模型路径: {output_dir}")
    return output_dir


class ONNXRerankProvider(RerankProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

        plugin_config = get_plugin_config()
        self.hf_mirror = provider_config.get(
            "huggingface_mirror", plugin_config.get("huggingface_mirror", "")
        )
        self.auto_download = (
            provider_config.get("auto_download", plugin_config.get("auto_download", 1))
            == 1
        )

        self.auto_unload_timeout = provider_config.get(
            "auto_unload_timeout", plugin_config.get("auto_unload_timeout", 0)
        )
        self._last_used_time: float = 0
        self._auto_unload_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        base_path = provider_config.get("ONNXRerank_path", DEFAULT_RERANK_MODEL)
        data_dir = Path(StarTools.get_data_dir("ONNXEmbedding"))
        base_path = Path(base_path)
        self.model_path = base_path if base_path.is_absolute() else data_dir / base_path

        self.tokenizer_path = provider_config.get(
            "ONNXRerank_tokenizer_path",
            str(self.model_path / "tokenizer.json"),
        )

        self.session = None
        self.tokenizer = None
        self._model_lock = asyncio.Lock()

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

        self.max_length = provider_config.get("ONNXRerank_max_length", 512)

        logger.info(
            f"[ONNXRerank] Provider 初始化完成，"
            f"env_available={self._env_available}, "
            f"model_path={self.model_path}"
        )

    def _ensure_env_available(self):
        if not self._env_available:
            raise RuntimeError(f"[ONNXRerank] 环境不可用: {self._env_error}")

    def _load_tokenizer(self) -> Any:
        from tokenizers import Tokenizer

        tokenizer_path = Path(self.tokenizer_path)
        if not tokenizer_path.is_absolute():
            tokenizer_path = (
                Path(StarTools.get_data_dir("ONNXEmbedding")) / tokenizer_path
            )

        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
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

        tokenizer.enable_truncation(max_length=self.max_length)
        return tokenizer

    def _resolve_model_file(self) -> tuple[Path, bool]:
        """Return (model_file_or_ir_dir, is_openvino_ir)."""
        model_path = Path(self.model_path)
        # OpenVINO IR: prefer openvino_model.xml inside the directory
        if model_path.is_dir():
            ir_xml = model_path / "openvino_model.xml"
            if ir_xml.exists():
                return ir_xml, True
            onnx_files = list(model_path.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(
                    f"在目录 {model_path} 中找不到 ONNX 或 IR 模型文件"
                )
            return onnx_files[0], False
        if model_path.suffix == ".xml" and model_path.exists():
            return model_path, True
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        return model_path, False

    def _load_onnx_session(self) -> Any:
        model_file, is_ir = self._resolve_model_file()
        backend = self.provider_config.get("ONNXRerank_backend", "auto")

        if backend in ("openvino", "auto"):
            try:
                import openvino as ov

                core = ov.Core()
                devices = core.available_devices
                # Prefer GPU, then NPU, then CPU
                preferred_device = "CPU"
                for dev in ("GPU", "NPU"):
                    if dev in devices:
                        preferred_device = dev
                        break

                # Hint for quantized (INT8) models: allow cache to avoid re-compilation
                ov_config: dict = {}
                model_dir = (
                    Path(self.model_path)
                    if Path(self.model_path).is_dir()
                    else Path(self.model_path).parent
                )
                cache_dir = model_dir / ".ov_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                core.set_property({"CACHE_DIR": str(cache_dir)})

                logger.info(
                    f"[ONNXRerank] OpenVINO 可用设备: {devices}, 使用: {preferred_device}"
                )
                compiled_model = core.compile_model(
                    str(model_file), preferred_device, ov_config
                )
                logger.info(
                    f"[ONNXRerank] OpenVINO 后端加载成功 "
                    f"({'IR' if is_ir else 'ONNX'} format, device={preferred_device})"
                )
                return ("openvino", compiled_model)
            except ImportError:
                if backend == "openvino":
                    raise RuntimeError("未安装 openvino，请执行：pip install openvino")
                logger.info("[ONNXRerank] OpenVINO 未安装，使用 ONNX Runtime")
            except Exception as exc:
                if backend == "openvino":
                    raise RuntimeError(f"OpenVINO 加载模型失败: {exc}") from exc
                logger.warning(
                    f"[ONNXRerank] OpenVINO 加载失败: {exc}，回退到 ONNX Runtime"
                )

        # ---- ONNX Runtime path ----
        import onnxruntime as ort

        optimization_level = self.provider_config.get(
            "ONNXRerank_optimization_level", "disable"
        )
        sess_options = ort.SessionOptions()
        _OPT_MAP = {
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        }
        sess_options.graph_optimization_level = _OPT_MAP.get(
            optimization_level, ort.GraphOptimizationLevel.ORT_DISABLE_ALL
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
        logger.info(f"[ONNXRerank] ONNX Runtime 提供程序: {selected_providers}")

        try:
            return (
                "onnxruntime",
                ort.InferenceSession(
                    str(model_file),
                    sess_options,
                    providers=selected_providers,
                ),
            )
        except Exception as exc:
            logger.warning(f"[ONNXRerank] 加载模型失败，尝试关闭优化重试: {exc}")
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            return (
                "onnxruntime",
                ort.InferenceSession(
                    str(model_file),
                    sess_options,
                    providers=selected_providers,
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

            logger.info(f"[ONNXRerank] 开始加载模型: {self.model_path}")
            loop = asyncio.get_running_loop()

            try:
                self.tokenizer = await loop.run_in_executor(None, self._load_tokenizer)
                self.session = await loop.run_in_executor(None, self._load_onnx_session)
                self._update_last_used_time()
                self._start_auto_unload_task()
                logger.info("[ONNXRerank] 模型加载成功")
            except Exception as e:
                logger.error("[ONNXRerank] 模型加载失败", exc_info=True)
                raise RuntimeError(f"模型加载失败: {e}") from e

    async def _download_model_if_needed(self):
        if self._check_model_exists():
            return

        if not self.auto_download:
            raise FileNotFoundError(
                f"Rerank模型文件不存在: {self.model_path}，且未启用自动下载。"
            )

        model_name = self._extract_model_name_from_path()
        if not model_name:
            raise FileNotFoundError(
                f"Rerank模型文件不存在: {self.model_path}，且无法从路径推断模型名称。"
            )

        logger.info(f"[ONNXRerank] 模型不存在，开始自动下载: {model_name}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _download_rerank_model_from_hf,
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
            logger.error("[ONNXRerank] 资源清理失败", exc_info=True)
            return False

    def _update_last_used_time(self):
        self._last_used_time = time.time()

    def _start_auto_unload_task(self):
        logger.info(f"[ONNXRerank] 自动卸载超时配置: {self.auto_unload_timeout} 分钟")
        if self.auto_unload_timeout <= 0:
            logger.info("[ONNXRerank] 自动卸载未启用（auto_unload_timeout <= 0）")
            return
        if self._auto_unload_task is not None and not self._auto_unload_task.done():
            return

        self._shutdown_event.clear()
        self._auto_unload_task = asyncio.create_task(self._auto_unload_loop())
        logger.info(
            f"[ONNXRerank] 已启动自动卸载任务，超时时间: {self.auto_unload_timeout} 分钟"
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
                        f"[ONNXRerank] 模型已 {self.auto_unload_timeout} 分钟未使用，自动卸载"
                    )
                    async with self._model_lock:
                        if self.session is not None:
                            self._cleanup_resources()
                            logger.info("[ONNXRerank] 模型已自动卸载")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ONNXRerank] 自动卸载检查出错: {e}")

    def _stop_auto_unload_task(self):
        if self._auto_unload_task is not None:
            self._shutdown_event.set()
            if not self._auto_unload_task.done():
                self._auto_unload_task.cancel()
            self._auto_unload_task = None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        await self._ensure_model_loaded()
        self._update_last_used_time()

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            self._compute_scores,
            query,
            documents,
        )

        results = [
            RerankResult(index=i, relevance_score=float(scores[i]))
            for i in range(len(documents))
        ]

        results.sort(key=lambda x: x.relevance_score, reverse=True)

        if top_n is not None:
            results = results[:top_n]

        return results

    def _compute_scores(self, query: str, documents: list[str]) -> np.ndarray:
        if self.tokenizer is None or self.session is None:
            raise RuntimeError("模型未加载")

        input_ids_list = []
        attention_mask_list = []

        for doc in documents:
            encoding = self.tokenizer.encode(query, doc)
            input_ids_list.append(encoding.ids)
            attention_mask_list.append(encoding.attention_mask)

        max_len = max(len(ids) for ids in input_ids_list)
        input_ids = np.zeros((len(documents), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(documents), max_len), dtype=np.int64)

        for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(mask)] = mask

        backend, model = self.session

        if backend == "openvino":
            return self._compute_scores_openvino(model, input_ids, attention_mask)
        else:
            return self._compute_scores_onnxruntime(model, input_ids, attention_mask)

    def _compute_scores_openvino(
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

        logits = list(infer_request.results.values())[0].data[:]

        if len(logits.shape) == 2:
            scores = logits[:, 0]
        else:
            scores = logits.flatten()

        scores = 1 / (1 + np.exp(-scores))
        return scores

    def _compute_scores_onnxruntime(
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

        logits = outputs[0]

        if len(logits.shape) == 2:
            scores = logits[:, 0]
        else:
            scores = logits.flatten()

        scores = 1 / (1 + np.exp(-scores))
        return scores

    async def unload_model(self) -> bool:
        self._stop_auto_unload_task()
        async with self._model_lock:
            if self.session is None:
                logger.info("[ONNXRerank] 模型未加载，无需卸载")
                return True

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._cleanup_resources)

    def force_unload_sync(self) -> bool:
        self._stop_auto_unload_task()
        if self.session is None:
            return True
        return self._cleanup_resources()


def register_ONNXRerankProvider() -> None:
    if "ONNXRerank" in provider_cls_map:
        logger.debug("[ONNXRerank] Provider 已存在，跳过注册")
        return
    try:
        register_provider_adapter(
            "ONNXRerank",
            "ONNX Runtime Rerank Provider",
            provider_type=ProviderType.RERANK,
        )(ONNXRerankProvider)
        logger.info("[ONNXRerank] Provider 已注册")
    except ValueError:
        logger.debug("[ONNXRerank] Provider 已存在（race），跳过")
