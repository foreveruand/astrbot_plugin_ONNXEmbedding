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

        self.backend: str = provider_config.get("ONNXChat_backend", "auto")
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
            f"backend={self.backend} device={self.device} "
            f"env_available={self._env_available} "
            f"available_backend={self._available_backend}"
        )

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
            logger.info(f"[ONNXChat] 模型加载成功 (backend={self._pipeline[0]})")

    def _load_pipeline(self) -> tuple:
        model_path_str = str(self.model_path)
        if self._available_backend == "openvino":
            return self._load_openvino_pipeline(model_path_str)
        elif self._available_backend == "onnxruntime":
            return self._load_ortgenai_pipeline(model_path_str)
        else:
            raise RuntimeError("[ONNXChat] 无可用推理后端")

    def _load_openvino_pipeline(self, model_path: str) -> tuple:
        import openvino_genai as ov_genai

        device = self.device
        try:
            pipeline = ov_genai.LLMPipeline(model_path, device)
            logger.info(f"[ONNXChat] OpenVINO GenAI 加载成功 device={device}")
            return ("openvino", pipeline)
        except Exception as exc:
            if device == "CPU":
                raise RuntimeError(
                    f"[ONNXChat] OpenVINO GenAI 加载失败 (device={device}): {exc}"
                ) from exc
            logger.warning(
                f"[ONNXChat] OpenVINO GenAI 加载失败 (device={device}): {exc}，回退到 CPU"
            )
            pipeline = ov_genai.LLMPipeline(model_path, "CPU")
            logger.info("[ONNXChat] OpenVINO GenAI (CPU fallback) 加载成功")
            return ("openvino", pipeline)

    def _load_ortgenai_pipeline(self, model_path: str) -> tuple:
        import onnxruntime_genai as ortg

        model = ortg.Model(model_path)
        tokenizer = ortg.Tokenizer(model)
        logger.info("[ONNXChat] ONNX Runtime GenAI 加载成功")
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
