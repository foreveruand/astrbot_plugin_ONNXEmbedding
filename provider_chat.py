"""
ONNX Chat Provider for AstrBot
基于ONNX Runtime的轻量级文本生成模型Provider
"""

import asyncio
import gc
import random
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
from astrbot.api import logger
from astrbot.core.provider.provider import Provider
from astrbot.core.provider.entities import LLMResponse, ProviderType
from astrbot.core.provider.register import (
    provider_registry,
    register_provider_adapter,
)
from astrbot.core.config.default import CONFIG_METADATA_2


class ONNXChatProvider(Provider):
    """ONNX Runtime based chat completion provider"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

        # 模型路径处理
        model_path = provider_config.get("ONNXChat_model_path", "")
        if not model_path:
            raise ValueError("ONNXChat_model_path 不能为空")

        from astrbot.api.star import StarTools

        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = Path(StarTools.get_data_dir()) / model_path

        self.model_path = model_path
        self.tokenizer_path = provider_config.get(
            "ONNXChat_tokenizer_path",
            str(self.model_path / "tokenizer.json"),
        )

        # 生成参数
        self.max_length = int(provider_config.get("ONNXChat_max_length", 256))
        self.max_new_tokens = int(provider_config.get("ONNXChat_max_new_tokens", 128))
        self.temperature = float(provider_config.get("ONNXChat_temperature", 0.8))
        self.top_p = float(provider_config.get("ONNXChat_top_p", 0.9))
        self.top_k = int(provider_config.get("ONNXChat_top_k", 50))

        # 运行状态
        self.session = None
        self.tokenizer = None
        self._model_lock = asyncio.Lock()

        # 环境检测
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
            f"[ONNXChat] Provider 初始化完成，"
            f"env_available={self._env_available}, "
            f"model_path={self.model_path}"
        )

    def _ensure_env_available(self):
        if not self._env_available:
            raise RuntimeError(f"[ONNXChat] 环境不可用: {self._env_error}")

    def _load_tokenizer(self) -> Any:
        """加载tokenizer"""
        from tokenizers import Tokenizer

        tokenizer_path = Path(self.tokenizer_path)
        if not tokenizer_path.is_absolute():
            from astrbot.api.star import StarTools

            tokenizer_path = Path(StarTools.get_data_dir()) / tokenizer_path

        if tokenizer_path.exists():
            return Tokenizer.from_file(str(tokenizer_path))

        # 尝试在模型目录中查找
        model_dir = Path(self.model_path)
        if model_dir.is_dir():
            for tokenizer_file in ["tokenizer.json"]:
                candidate = model_dir / tokenizer_file
                if candidate.exists():
                    return Tokenizer.from_file(str(candidate))

        raise FileNotFoundError(f"找不到tokenizer文件: {tokenizer_path}")

    def _load_onnx_session(self) -> Any:
        """加载ONNX模型session"""
        import onnxruntime as ort

        model_path = Path(self.model_path)

        # 如果路径是目录，查找.onnx文件
        if model_path.is_dir():
            onnx_files = list(model_path.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"在目录 {model_path} 中找不到ONNX模型文件")
            model_path = onnx_files[0]

        if not model_path.exists():
            raise FileNotFoundError(f"找不到ONNX模型文件: {model_path}")

        # 配置ONNX Runtime会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

        # 自动选择执行提供程序
        providers = ort.get_available_providers()
        preferred_order = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected_providers = [p for p in preferred_order if p in providers]

        logger.info(f"[ONNXChat] 使用执行提供程序: {selected_providers}")

        return ort.InferenceSession(
            str(model_path), sess_options, providers=selected_providers
        )

    async def _ensure_model_loaded(self):
        """Lazy Loading + 并发安全"""
        self._ensure_env_available()

        if self.session is not None and self.tokenizer is not None:
            return

        async with self._model_lock:
            if self.session is not None and self.tokenizer is not None:
                return

            logger.info(f"[ONNXChat] 开始加载模型: {self.model_path}")
            loop = asyncio.get_running_loop()

            try:
                self.tokenizer = await loop.run_in_executor(None, self._load_tokenizer)
                self.session = await loop.run_in_executor(None, self._load_onnx_session)
                logger.info("[ONNXChat] 模型加载成功")
            except Exception as e:
                logger.error("[ONNXChat] 模型加载失败", exc_info=True)
                raise RuntimeError(f"模型加载失败: {e}") from e

    def _cleanup_resources(self) -> bool:
        """统一的模型 / 显存 / 内存清理逻辑"""
        try:
            self.session = None
            self.tokenizer = None
            gc.collect()
            return True
        except Exception:
            logger.error("[ONNXChat] 资源清理失败", exc_info=True)
            return False

    async def unload_model(self) -> bool:
        async with self._model_lock:
            if self.session is None:
                logger.info("[ONNXChat] 模型未加载，无需卸载")
                return True

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._cleanup_resources)

    def get_model(self) -> str:
        """获取当前模型名称"""
        return str(self.provider_config.get("ONNXChat_model_name", "onnx-chat-model"))

    def set_model(self, model: str) -> None:
        """设置模型名称"""
        self.provider_config["ONNXChat_model_name"] = model

    def get_max_token(self) -> int:
        """获取最大token数"""
        return self.max_new_tokens

    def get_temperature(self) -> float:
        """获取温度参数"""
        return self.temperature

    def get_top_p(self) -> float:
        """获取top_p参数"""
        return self.top_p

    def _sample_token(self, logits: np.ndarray) -> int:
        """从logits中采样下一个token"""
        # 应用温度
        logits = logits / self.temperature

        # 应用top-k
        if self.top_k > 0:
            indices_to_remove = logits < np.partition(logits, -self.top_k)[-self.top_k]
            logits[indices_to_remove] = -float("Inf")

        # 应用top-p (nucleus sampling)
        sorted_logits = np.sort(logits)[::-1]
        sorted_indices = np.argsort(logits)[::-1]
        cumulative_probs = np.cumsum(
            np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
        )

        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # 采样
        return int(np.random.choice(len(probs), p=probs))

    async def _generate_tokens(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> AsyncGenerator[str, None]:
        """生成token序列"""
        if self.session is None or self.tokenizer is None:
            raise RuntimeError("模型未加载")

        generated_tokens = []
        current_ids = input_ids.copy()
        current_mask = attention_mask.copy()

        input_names = [inp.name for inp in self.session.get_inputs()]

        for _ in range(self.max_new_tokens):
            # 构建输入
            feed_dict = {}
            if "input_ids" in input_names:
                feed_dict["input_ids"] = current_ids
            if "attention_mask" in input_names:
                feed_dict["attention_mask"] = current_mask

            # 运行推理
            outputs = self.session.run(None, feed_dict)
            logits = outputs[0][:, -1, :]  # 取最后一个位置的logits

            # 采样下一个token
            next_token_id = self._sample_token(logits[0])
            generated_tokens.append(next_token_id)

            # 解码并yield
            token_text = self.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )
            if token_text:
                yield token_text

            # 更新输入
            current_ids = np.concatenate(
                [current_ids, np.array([[next_token_id]], dtype=np.int64)], axis=1
            )
            current_mask = np.concatenate(
                [current_mask, np.array([[1]], dtype=np.int64)], axis=1
            )

            # 检查是否生成结束
            if len(current_ids[0]) >= self.max_length:
                break

            # 检查是否生成了结束token (假设结束token id为2)
            if next_token_id == 2:
                break

    async def text_chat(
        self,
        prompt: str,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        contexts: list[dict] | list[Any] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """同步聊天接口"""
        await self._ensure_model_loaded()

        # 构建输入
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n"
        if contexts:
            for ctx in contexts:
                if isinstance(ctx, dict):
                    role = ctx.get("role", "user")
                    content = ctx.get("content", "")
                    full_prompt += f"{role}: {content}\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        # Tokenize
        encoding = self.tokenizer.encode(full_prompt)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        # 截断
        if len(input_ids[0]) > self.max_length:
            input_ids = input_ids[:, -self.max_length :]
            attention_mask = attention_mask[:, -self.max_length :]

        # 生成
        generated_text = ""
        async for token in self._generate_tokens(input_ids, attention_mask):
            generated_text += token

        # 构建响应
        response = LLMResponse()
        response.completion_text = generated_text
        response.role = "assistant"
        response.model = self.get_model()
        return response

    async def text_chat_stream(
        self,
        prompt: str,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        contexts: list[dict] | list[Any] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: Any | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """流式聊天接口"""
        await self._ensure_model_loaded()

        # 构建输入
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n"
        if contexts:
            for ctx in contexts:
                if isinstance(ctx, dict):
                    role = ctx.get("role", "user")
                    content = ctx.get("content", "")
                    full_prompt += f"{role}: {content}\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        # Tokenize
        encoding = self.tokenizer.encode(full_prompt)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        # 截断
        if len(input_ids[0]) > self.max_length:
            input_ids = input_ids[:, -self.max_length :]
            attention_mask = attention_mask[:, -self.max_length :]

        # 流式生成
        generated_text = ""
        async for token in self._generate_tokens(input_ids, attention_mask):
            generated_text += token
            response = LLMResponse()
            response.completion_text = token
            response.role = "assistant"
            response.model = self.get_model()
            yield response

        # 最终响应
        final_response = LLMResponse()
        final_response.completion_text = generated_text
        final_response.role = "assistant"
        final_response.model = self.get_model()
        yield final_response


# ============================================================
# Provider 注册函数
# ============================================================
def register_ONNXChatProvider():
    try:
        register_provider_adapter(
            "ONNXChat",
            "ONNX Runtime Chat Completion Provider",
            provider_type=ProviderType.CHAT_COMPLETION,
        )(ONNXChatProvider)
        logger.info("[ONNXChat] Provider 已注册")
    except ValueError:
        logger.info("[ONNXChat] Provider 已存在，跳过注册")
