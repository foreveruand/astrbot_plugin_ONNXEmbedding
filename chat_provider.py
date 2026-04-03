"""Local quantized LLM Chat Provider for AstrBot.

Uses **OpenVINO GenAI** as the primary backend (recommended for Intel hardware)
and falls back to **ONNX Runtime GenAI** when openvino_genai is not installed.

Supported model formats
-----------------------
* OpenVINO IR (.xml + .bin) — export via ``optimum-intel`` or OpenVINO Model Optimizer
* ONNX Runtime GenAI (.onnx) — export via ``optimum`` with ``--task text-generation``

Quantized models (INT8/FP16) from `Hugging Face <https://huggingface.co/Intel>`_
work out of the box with OpenVINO GenAI.

Config template keys (set in the provider's config dict)
---------------------------------------------------------
``ONNXChat_path``          : str  — local path or HF model slug
``ONNXChat_backend``       : str  — "auto" | "openvino" | "onnxruntime"
``ONNXChat_device``        : str  — "CPU" | "GPU" | "AUTO" (OpenVINO only)
``ONNXChat_max_new_tokens``: int  — maximum tokens to generate (default 512)
``ONNXChat_temperature``   : float — sampling temperature (default 0.7; 0 = greedy)
``ONNXChat_context_length``: int  — context window size hint (default 2048)
``auto_download``          : int  — 1 to auto-download from HuggingFace
``huggingface_mirror``     : str  — optional mirror URL
"""

from __future__ import annotations

import asyncio
import gc
import json
import os  # noqa: F401 (required by openvino_genai)
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal

from astrbot.api import logger
from astrbot.core.agent.message import ContentPart, Message
from astrbot.core.agent.tool import ToolSet
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, ProviderType, ToolCallsResult
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.register import provider_cls_map, register_provider_adapter
from astrbot.core.star import StarTools

from ._plugin_config import get_plugin_config
from .model_store import check_model_exists, download_chat_model

DEFAULT_CHAT_MODEL = "microsoft/phi-2"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONTEXT_LENGTH = 2048


class ONNXChatProvider(Provider):
    """Local quantized LLM backed by OpenVINO GenAI or ONNX Runtime GenAI."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

        plugin_cfg = get_plugin_config()

        self.hf_mirror: str = provider_config.get(
            "huggingface_mirror",
            plugin_cfg.get("huggingface_mirror", ""),
        )
        self.auto_download: bool = (
            provider_config.get("auto_download", plugin_cfg.get("auto_download", 0))
            == 1
        )

        base_path = provider_config.get("ONNXChat_path", DEFAULT_CHAT_MODEL)
        data_dir = Path(StarTools.get_data_dir("ONNXEmbedding"))
        base_path = Path(base_path)
        self.model_path: Path = (
            base_path if base_path.is_absolute() else data_dir / base_path
        )

        self.requested_backend: str = str(
            provider_config.get("ONNXChat_backend", "auto")
        )
        self.backend: str = self._normalize_backend(self.requested_backend)
        self.device: str = provider_config.get("ONNXChat_device", "CPU")
        self.max_new_tokens: int = int(
            provider_config.get("ONNXChat_max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        )
        self.temperature: float = float(
            provider_config.get("ONNXChat_temperature", DEFAULT_TEMPERATURE)
        )
        self.context_length: int = int(
            provider_config.get("ONNXChat_context_length", DEFAULT_CONTEXT_LENGTH)
        )

        # Runtime state
        self._pipeline: tuple | None = None  # ("openvino"|"onnxruntime", ...) tuple
        self._model_lock = asyncio.Lock()

        # Backend availability
        self._env_available: bool = True
        self._env_error: str | None = None
        self._available_backend: str | None = None
        self._probe_backends()

        model_label = provider_config.get("ONNXChat_path", DEFAULT_CHAT_MODEL)
        self.set_model(model_label)

        logger.info(
            f"[ONNXChat] Provider 初始化完成 model_path={self.model_path} "
            f"backend={self.requested_backend}->{self.backend} device={self.device} "
            f"env_available={self._env_available} "
            f"available_backend={self._available_backend}"
        )

    @staticmethod
    def _normalize_backend(value: str | None) -> str:
        """Normalize backend aliases from config/UI values."""
        raw = str(value or "auto").strip().lower()
        alias_map = {
            "auto": "auto",
            "openvino": "openvino",
            "ov": "openvino",
            "onnx": "onnxruntime",
            "ort": "onnxruntime",
            "onnxruntime": "onnxruntime",
            "onnx-runtime": "onnxruntime",
        }
        return alias_map.get(raw, raw or "auto")

    @staticmethod
    def _can_import_backend(backend: str) -> bool:
        """Return True if the requested backend module can be imported."""
        try:
            if backend == "openvino":
                import openvino_genai  # noqa: F401
            elif backend == "onnxruntime":
                import onnxruntime_genai  # noqa: F401
            else:
                return False
            return True
        except ImportError:
            return False

    def _probe_backends(self) -> None:
        """Detect which inference backend is installed at init time."""
        want_ov = self.backend in ("openvino", "auto")
        want_ort = self.backend in ("onnxruntime", "auto")

        if want_ov:
            try:
                import openvino_genai  # noqa: F401

                self._available_backend = "openvino"
                return
            except ImportError:
                if self.backend == "openvino":
                    self._env_available = False
                    self._env_error = (
                        "未安装 openvino-genai，请执行：pip install openvino-genai"
                    )
                    return
                logger.debug("[ONNXChat] openvino_genai 未安装，尝试 onnxruntime-genai")

        if want_ort:
            try:
                import onnxruntime_genai  # noqa: F401

                self._available_backend = "onnxruntime"
                return
            except ImportError:
                pass

        if self._available_backend is None:
            self._env_available = False
            if self.backend == "openvino":
                self._env_error = (
                    "未安装 openvino-genai，请在 AstrBot 当前运行环境中执行："
                    "pip install openvino-genai"
                )
            elif self.backend == "onnxruntime":
                self._env_error = (
                    "未安装 onnxruntime-genai，请在 AstrBot 当前运行环境中执行："
                    "pip install onnxruntime-genai"
                )
            else:
                self._env_error = (
                    "未安装本地推理后端，请安装以下任一套件：\n"
                    "  pip install openvino-genai      # Intel GPU/CPU 推荐\n"
                    "  pip install onnxruntime-genai   # 通用（CPU/CUDA）"
                )

    # ------------------------------------------------------------------
    # Provider abstract-method implementations
    # ------------------------------------------------------------------

    def get_current_key(self) -> str:
        return ""  # local models have no API key

    def set_key(self, key: str) -> None:
        pass  # no-op

    async def get_models(self) -> list[str]:
        return [self.provider_config.get("ONNXChat_path", DEFAULT_CHAT_MODEL)]

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    async def _ensure_model_loaded(self) -> None:
        if not self._env_available:
            raise RuntimeError(f"[ONNXChat] 环境不可用: {self._env_error}")

        if self._pipeline is not None:
            return

        async with self._model_lock:
            if self._pipeline is not None:
                return

            if not check_model_exists(self.model_path):
                if not self.auto_download:
                    raise FileNotFoundError(
                        f"[ONNXChat] 模型不存在: {self.model_path}。"
                        "请手动下载或将 auto_download 设置为 1。"
                    )
                model_name = self.provider_config.get(
                    "ONNXChat_path", DEFAULT_CHAT_MODEL
                )
                logger.info(f"[ONNXChat] 模型不存在，开始自动下载: {model_name}")
                loop = asyncio.get_running_loop()
                ok, _ = await loop.run_in_executor(
                    None,
                    download_chat_model,
                    model_name,
                    self.model_path,
                    self.hf_mirror,
                    self._available_backend or "auto",
                )
                if not ok:
                    raise RuntimeError(f"[ONNXChat] 模型下载失败: {model_name}")

            logger.info(
                f"[ONNXChat] 开始加载模型: {self.model_path} "
                f"(backend={self._available_backend}, device={self.device})"
            )
            loop = asyncio.get_running_loop()
            self._pipeline = await loop.run_in_executor(None, self._load_pipeline)
            pipeline = self._pipeline
            if pipeline is None:
                raise RuntimeError("[ONNXChat] 模型加载失败：未返回有效的 pipeline")
            logger.info(f"[ONNXChat] 模型加载成功 (backend={pipeline[0]})")

    def _load_pipeline(self) -> tuple:
        resolved = self._resolve_model_dir()
        model_path_str = str(resolved)
        has_openvino = any(resolved.glob("*.xml"))
        has_onnx = any(resolved.glob("*.onnx"))

        selected_backend = self.backend
        if selected_backend == "auto":
            if has_openvino and self._can_import_backend("openvino"):
                selected_backend = "openvino"
            elif has_onnx and self._can_import_backend("onnxruntime"):
                selected_backend = "onnxruntime"
            else:
                selected_backend = self._available_backend or "auto"
        elif selected_backend == "openvino" and not has_openvino and has_onnx:
            if self._can_import_backend("onnxruntime"):
                logger.warning(
                    "[ONNXChat] 当前目录只有 ONNX 文件，自动切换到 onnxruntime 后端"
                )
                selected_backend = "onnxruntime"
        elif selected_backend == "onnxruntime" and not has_onnx and has_openvino:
            if self._can_import_backend("openvino"):
                logger.warning(
                    "[ONNXChat] 当前目录只有 OpenVINO IR 文件，自动切换到 openvino 后端"
                )
                selected_backend = "openvino"

        logger.info(
            f"[ONNXChat] 实际模型目录: {model_path_str} "
            f"(requested={self.requested_backend}, normalized={self.backend}, selected={selected_backend})"
        )

        if selected_backend == "openvino":
            if not has_openvino:
                raise RuntimeError(
                    "[ONNXChat] 目录中未找到 OpenVINO IR 模型文件 (*.xml/*.bin)"
                )
            return self._load_openvino_pipeline(model_path_str)
        if selected_backend == "onnxruntime":
            if not has_onnx:
                raise RuntimeError("[ONNXChat] 目录中未找到 ONNX 模型文件 (*.onnx)")
            return self._load_ortgenai_pipeline(model_path_str)

        raise RuntimeError("[ONNXChat] 无可用推理后端")

    def _resolve_model_dir(self) -> Path:
        """Return the directory that actually contains model weight files.

        Quantized model repos often place weights in a sub-folder (e.g. ``onnx/``,
        ``openvino/``).  This method searches up to two levels deep and returns
        the first directory that contains a ``.xml``, ``.bin``, or ``.onnx`` file.
        Falls back to ``self.model_path`` if nothing is found.
        """
        root = self.model_path
        # Check root first
        if self._dir_has_model_files(root):
            return root
        # Check one level of subdirectories
        if root.is_dir():
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and self._dir_has_model_files(sub):
                    logger.info(f"[ONNXChat] 在子目录找到模型文件: {sub}")
                    return sub
        return root

    @staticmethod
    def _dir_has_model_files(directory: Path) -> bool:
        """Return True if *directory* contains any .xml, .onnx, or .bin file."""
        if not directory.is_dir():
            return False
        for ext in ("*.xml", "*.onnx", "*.bin"):
            if any(directory.glob(ext)):
                return True
        return False

    @staticmethod
    def _first_existing_relative_path(root: Path, patterns: list[str]) -> str | None:
        """Return the first file path matching *patterns*, relative to *root*."""
        for pattern in patterns:
            for candidate in sorted(root.rglob(pattern)):
                if candidate.is_file():
                    return str(candidate.relative_to(root))
        return None

    def _ensure_genai_config(self, model_root: Path) -> Path:
        """Ensure a usable `genai_config.json` exists for ORT GenAI models."""
        model_root = Path(model_root)
        config_path = model_root / "genai_config.json"
        if config_path.exists():
            return config_path

        genai_config = self._build_genai_config(model_root)
        if genai_config is None:
            raise RuntimeError(
                "[ONNXChat] 当前 ONNX 模型缺少 genai_config.json，且无法自动生成。"
                "请使用 ONNX Runtime GenAI 导出的模型目录，或改用 OpenVINO 模型。"
            )

        config_path.write_text(
            json.dumps(genai_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"[ONNXChat] 已自动生成 genai_config.json: {config_path}")
        return config_path

    def _build_genai_config(self, model_root: Path) -> dict | None:
        """Build a minimal ORT GenAI config from HF `config.json` files.

        This is primarily for repos like `onnx-community/Qwen3.5-0.8B-ONNX`
        which contain valid ONNX weights but do not ship a prebuilt
        `genai_config.json`.
        """
        config_file = model_root / "config.json"
        if not config_file.exists():
            return None

        try:
            raw_cfg = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"[ONNXChat] 读取 config.json 失败: {exc}")
            return None

        generation_cfg: dict[str, Any] = {}
        generation_config_file = model_root / "generation_config.json"
        if generation_config_file.exists():
            try:
                generation_cfg = json.loads(
                    generation_config_file.read_text(encoding="utf-8")
                )
            except Exception:
                generation_cfg = {}

        text_cfg = raw_cfg.get("text_config") or raw_cfg
        model_type = str(
            raw_cfg.get("model_type") or text_cfg.get("model_type") or "generic"
        ).lower()

        decoder_filename = self._first_existing_relative_path(
            model_root,
            [
                "decoder_model_merged*.onnx",
                "decoder*.onnx",
                "model*.onnx",
            ],
        )
        if not decoder_filename:
            return None

        embed_filename = self._first_existing_relative_path(
            model_root,
            ["embed_tokens*.onnx", "embedding*.onnx"],
        )
        vision_filename = self._first_existing_relative_path(
            model_root,
            ["vision_encoder*.onnx", "vision*.onnx"],
        )

        hidden_size = int(
            text_cfg.get("hidden_size") or raw_cfg.get("hidden_size") or 0
        )
        num_heads = int(
            text_cfg.get("num_attention_heads")
            or raw_cfg.get("num_attention_heads")
            or 0
        )
        num_layers = int(
            text_cfg.get("num_hidden_layers") or raw_cfg.get("num_hidden_layers") or 0
        )
        num_kv_heads = int(
            text_cfg.get("num_key_value_heads")
            or raw_cfg.get("num_key_value_heads")
            or num_heads
            or 0
        )
        vocab_size = int(text_cfg.get("vocab_size") or raw_cfg.get("vocab_size") or 0)
        context_length = int(
            text_cfg.get("max_position_embeddings")
            or raw_cfg.get("max_position_embeddings")
            or self.context_length
        )
        head_size = hidden_size // num_heads if hidden_size and num_heads else 0

        eos_token_id = (
            generation_cfg.get("eos_token_id")
            or text_cfg.get("eos_token_id")
            or raw_cfg.get("eos_token_id")
            or 1
        )
        if not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]

        bos_token_id = (
            generation_cfg.get("bos_token_id")
            or text_cfg.get("bos_token_id")
            or raw_cfg.get("bos_token_id")
        )
        if bos_token_id is None:
            bos_token_id = eos_token_id[0] if eos_token_id else 1

        pad_token_id = (
            generation_cfg.get("pad_token_id")
            or text_cfg.get("pad_token_id")
            or raw_cfg.get("pad_token_id")
        )
        if pad_token_id is None:
            pad_token_id = bos_token_id

        decoder_inputs = {
            ("inputs_embeds" if embed_filename else "input_ids"): (
                "inputs_embeds" if embed_filename else "input_ids"
            ),
            "attention_mask": "attention_mask",
            "position_ids": "position_ids",
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        }

        model_section: dict[str, Any] = {
            "bos_token_id": bos_token_id,
            "context_length": context_length,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": decoder_filename,
                "head_size": head_size,
                "hidden_size": hidden_size,
                "inputs": decoder_inputs,
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "num_attention_heads": num_heads,
                "num_hidden_layers": num_layers,
                "num_key_value_heads": num_kv_heads,
            },
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "type": raw_cfg.get("model_type") or model_type,
            "vocab_size": vocab_size,
        }

        if embed_filename:
            model_section["embedding"] = {
                "filename": embed_filename,
                "inputs": {
                    "input_ids": "input_ids",
                    "image_features": "image_features",
                },
                "outputs": {"inputs_embeds": "inputs_embeds"},
            }

        if vision_filename:
            vision_cfg = raw_cfg.get("vision_config") or {}
            model_section["vision"] = {
                "filename": vision_filename,
                "spatial_merge_size": int(vision_cfg.get("spatial_merge_size") or 2),
                "patch_size": int(vision_cfg.get("patch_size") or 14),
                "inputs": {
                    "pixel_values": "pixel_values",
                    "image_grid_thw": "image_grid_thw",
                },
                "outputs": {"image_features": "image_features"},
            }

        for token_key in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            if raw_cfg.get(token_key) is not None:
                model_section[token_key] = raw_cfg[token_key]

        search_section = {
            "diversity_penalty": float(generation_cfg.get("diversity_penalty", 0.0)),
            "do_sample": bool(generation_cfg.get("do_sample", self.temperature > 0)),
            "early_stopping": True,
            "length_penalty": float(generation_cfg.get("length_penalty", 1.0)),
            "max_length": int(generation_cfg.get("max_length", context_length)),
            "min_length": int(generation_cfg.get("min_length", 0)),
            "no_repeat_ngram_size": int(generation_cfg.get("no_repeat_ngram_size", 0)),
            "num_beams": int(generation_cfg.get("num_beams", 1)),
            "num_return_sequences": int(generation_cfg.get("num_return_sequences", 1)),
            "past_present_share_buffer": False,
            "repetition_penalty": float(generation_cfg.get("repetition_penalty", 1.0)),
            "temperature": float(generation_cfg.get("temperature", 1.0)),
            "top_k": int(generation_cfg.get("top_k", 50)),
            "top_p": float(generation_cfg.get("top_p", 1.0)),
        }

        return {"model": model_section, "search": search_section}

    def _load_openvino_pipeline(self, model_path: str) -> tuple:
        import openvino_genai as ov_genai

        def _try_load(target_path: str, target_device: str):
            pipeline_obj = ov_genai.LLMPipeline(target_path, target_device)
            try:
                tokenizer = pipeline_obj.get_tokenizer()
                chat_template = getattr(tokenizer, "chat_template", None)
                if chat_template:
                    tokenizer.set_chat_template(chat_template)
                    logger.info(
                        "[ONNXChat] 已按 OpenVINO/HuggingFace 推荐方式启用 chat template"
                    )
            except Exception as exc:
                logger.debug(
                    f"[ONNXChat] 设置 OpenVINO chat template 失败，继续使用默认提示词: {exc}"
                )
            logger.info(f"[ONNXChat] OpenVINO GenAI 加载成功 device={target_device}")
            return ("openvino", pipeline_obj)

        def _should_refresh_download(exc: Exception) -> bool:
            text = str(exc)
            markers = (
                "Incorrect weights in bin file",
                "Could not find a model in the directory",
                "Error opening",
            )
            return any(marker in text for marker in markers)

        device = self.device
        try:
            return _try_load(model_path, device)
        except Exception as exc:
            model_name = str(
                self.provider_config.get("ONNXChat_path", DEFAULT_CHAT_MODEL)
            )
            if (
                self.auto_download
                and not Path(model_name).is_absolute()
                and _should_refresh_download(exc)
            ):
                logger.warning(
                    "[ONNXChat] 检测到 OpenVINO 模型文件损坏或不完整，尝试强制重新下载..."
                )
                ok, failed = download_chat_model(
                    model_name,
                    self.model_path,
                    self.hf_mirror,
                    "openvino",
                    True,
                )
                if ok:
                    refreshed_path = str(self._resolve_model_dir())
                    try:
                        return _try_load(refreshed_path, device)
                    except Exception as retry_exc:
                        exc = retry_exc
                else:
                    logger.warning(f"[ONNXChat] 强制重新下载失败，缺失文件: {failed}")

            if device == "CPU":
                raise RuntimeError(
                    f"[ONNXChat] OpenVINO GenAI 加载失败 (device={device}): {exc}"
                ) from exc
            logger.warning(
                f"[ONNXChat] OpenVINO GenAI 加载失败 (device={device}): {exc}，回退到 CPU"
            )
            return _try_load(model_path, "CPU")

    def _load_ortgenai_pipeline(self, model_path: str) -> tuple:
        import onnxruntime_genai as ortg

        requested_root = Path(model_path)
        config_root = requested_root
        if (
            not (config_root / "config.json").exists()
            and (config_root.parent / "config.json").exists()
        ):
            config_root = config_root.parent

        self._ensure_genai_config(config_root)

        model = ortg.Model(str(config_root))
        tokenizer = ortg.Tokenizer(model)
        logger.info(f"[ONNXChat] ONNX Runtime GenAI 加载成功 root={config_root}")
        return ("onnxruntime", model, tokenizer)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        contexts: list[dict] | None,
        prompt: str | None,
        system_prompt: str | None,
    ) -> str:
        """Assemble a plain-text prompt from AstrBot context dicts."""
        parts: list[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")

        for msg in contexts or []:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content – extract text blocks only
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        if prompt:
            parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Synchronous inference helpers (run in executor)
    # ------------------------------------------------------------------

    def _run_openvino_inference(self, pipeline_obj: Any, prompt: str) -> str:
        try:
            import openvino_genai as ov_genai

            gen_config = ov_genai.GenerationConfig()
            gen_config.max_new_tokens = self.max_new_tokens
            if self.temperature > 0:
                gen_config.temperature = self.temperature
                gen_config.do_sample = True
            return pipeline_obj.generate(prompt, gen_config)
        except Exception:
            # Older API: pass kwargs directly
            return pipeline_obj.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

    def _run_ortgenai_inference(self, model: Any, tokenizer: Any, prompt: str) -> str:
        import onnxruntime_genai as ortg

        tokens = tokenizer.encode(prompt)
        params = ortg.GeneratorParams(model)
        params.max_length = len(tokens) + self.max_new_tokens
        params.input_ids = tokens
        if self.temperature > 0:
            params.set_search_options(do_sample=True, temperature=self.temperature)

        generator = ortg.Generator(model, params)
        output_tokens: list[int] = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            output_tokens.append(generator.get_next_tokens()[0])

        return tokenizer.decode(output_tokens)

    # ------------------------------------------------------------------
    # Provider public API
    # ------------------------------------------------------------------

    async def text_chat(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        func_tool: ToolSet | None = None,
        contexts: list[Message] | list[dict] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
        **kwargs,
    ) -> LLMResponse:
        await self._ensure_model_loaded()

        ctx_dicts = self._ensure_message_to_dicts(contexts)
        text_prompt = self._build_prompt(ctx_dicts, prompt, system_prompt)

        loop = asyncio.get_running_loop()
        backend_tag = self._pipeline[0]  # type: ignore[index]

        if backend_tag == "openvino":
            _, pipeline_obj = self._pipeline  # type: ignore[misc]
            completion = await loop.run_in_executor(
                None, self._run_openvino_inference, pipeline_obj, text_prompt
            )
        else:
            _, model_obj, tokenizer_obj = self._pipeline  # type: ignore[misc]
            completion = await loop.run_in_executor(
                None,
                self._run_ortgenai_inference,
                model_obj,
                tokenizer_obj,
                text_prompt,
            )

        result_chain = MessageChain().message(completion)
        return LLMResponse(role="assistant", result_chain=result_chain)

    async def text_chat_stream(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        func_tool: ToolSet | None = None,
        contexts: list[Message] | list[dict] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        # Non-streaming fallback: yield a single complete response
        result = await self.text_chat(
            prompt=prompt,
            session_id=session_id,
            image_urls=image_urls,
            func_tool=func_tool,
            contexts=contexts,
            system_prompt=system_prompt,
            tool_calls_result=tool_calls_result,
            model=model,
            **kwargs,
        )
        yield result

    async def test(self) -> None:
        """Quick availability test with extended timeout for cold-start model loading."""
        await asyncio.wait_for(
            self.text_chat(prompt="Reply PONG only."),
            timeout=120.0,
        )

    async def terminate(self) -> None:
        async with self._model_lock:
            if self._pipeline is not None:
                try:
                    self._pipeline = None
                    gc.collect()
                    logger.info("[ONNXChat] 模型已卸载")
                except Exception as exc:
                    logger.error(f"[ONNXChat] 卸载失败: {exc}")


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_ONNXChatProvider() -> None:
    if "ONNXChatProvider" in provider_cls_map:
        logger.debug("[ONNXChat] Provider 已存在，跳过注册")
        return
    try:
        register_provider_adapter(
            "ONNXChatProvider",
            "ONNX/OpenVINO 本地量化 Chat 模型 Provider",
            provider_type=ProviderType.CHAT_COMPLETION,
        )(ONNXChatProvider)
        logger.info("[ONNXChat] Provider 已注册")
    except ValueError:
        logger.debug("[ONNXChat] Provider 已存在（race），跳过")
