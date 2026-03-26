"""
ONNX Embedding Provider for AstrBot
基于ONNX Runtime的轻量级嵌入向量生成插件
"""

import asyncio
import gc
from pathlib import Path
from typing import Any

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

# 导入 Chat Provider
from .provider_chat import ONNXChatProvider, register_ONNXChatProvider

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_ONNX_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"


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


# ============================================================
# Embedding Provider
# ============================================================
class ONNXEmbeddingProvider(EmbeddingProvider):
    """ONNX Runtime based embedding provider"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)

        # -------- 模型路径处理（Pathlib）--------
        base_path = provider_config.get("ONNXEmbedding_path", DEFAULT_MODEL_NAME)

        data_dir = Path(StarTools.get_data_dir())
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
            tokenizer_path = Path(StarTools.get_data_dir()) / tokenizer_path

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
        """加载ONNX模型session"""
        import onnxruntime as ort

        model_path = Path(self.model_path)

        # 如果路径是目录，查找model.onnx文件
        if model_path.is_dir():
            onnx_files = list(model_path.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"在目录 {model_path} 中找不到ONNX模型文件")
            model_path = onnx_files[0]

        if not model_path.exists():
            raise FileNotFoundError(f"找不到ONNX模型文件: {model_path}")

        # 配置ONNX Runtime会话
        sess_options = ort.SessionOptions()

        # 根据配置决定是否启用图优化
        # 某些模型（特别是经过LayerNorm优化的）可能与图优化器不兼容
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
            # 默认禁用图优化，避免节点名称不匹配问题
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )

        logger.info(
            f"[ONNXEmbedding] 图优化级别: {optimization_level} "
            f"({sess_options.graph_optimization_level})"
        )

        # 自动选择执行提供程序
        providers = ort.get_available_providers()
        preferred_order = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DirectMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected_providers = [p for p in preferred_order if p in providers]

        logger.info(f"[ONNXEmbedding] 使用执行提供程序: {selected_providers}")

        # 尝试禁用特定的优化器来避免兼容性问题
        disabled_optimizers = None
        if optimization_level == "disable":
            # 禁用可能导致问题的 LayerNorm 和 Attention 优化器
            disabled_optimizers = [
                "SimplifiedLayerNormFusion",
                "LayerNormFusion",
                "AttentionFusion",
                "BiasSoftmaxFusion",
            ]
            logger.info(f"[ONNXEmbedding] 禁用优化器: {disabled_optimizers}")

        try:
            return ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=selected_providers,
                disabled_optimizers=disabled_optimizers,
            )
        except Exception as e:
            # 如果失败，尝试不使用任何优化
            logger.warning(f"[ONNXEmbedding] 加载模型失败，尝试不优化加载: {e}")
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            return ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=selected_providers,
                disabled_optimizers=[
                    "SimplifiedLayerNormFusion",
                    "LayerNormFusion",
                ],
            )

    async def _ensure_model_loaded(self):
        """
        Lazy Loading + 并发安全
        """
        self._ensure_env_available()

        if self.session is not None and self.tokenizer is not None:
            return

        async with self._model_lock:
            if self.session is not None and self.tokenizer is not None:
                return

            logger.info(f"[ONNXEmbedding] 开始加载模型: {self.model_path}")
            loop = asyncio.get_running_loop()

            try:
                # 加载tokenizer和session
                self.tokenizer = await loop.run_in_executor(None, self._load_tokenizer)
                self.session = await loop.run_in_executor(None, self._load_onnx_session)
                logger.info("[ONNXEmbedding] 模型加载成功")
            except Exception as e:
                logger.error("[ONNXEmbedding] 模型加载失败", exc_info=True)
                raise RuntimeError(f"模型加载失败: {e}") from e

    def _cleanup_resources(self) -> bool:
        """
        统一的模型 / 显存 / 内存清理逻辑
        """
        try:
            self.session = None
            self.tokenizer = None
            gc.collect()
            return True
        except Exception:
            logger.error("[ONNXEmbedding] 资源清理失败", exc_info=True)
            return False

    # ====================================================
    # Embedding API
    # ====================================================

    async def get_embedding(self, text: str) -> list[float]:
        """获取单个文本的嵌入向量"""
        await self._ensure_model_loaded()

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self._encode, [text])
        return embedding[0].tolist()

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的嵌入向量"""
        await self._ensure_model_loaded()

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self._encode, texts)
        return embeddings.tolist()

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        对文本列表进行编码，返回numpy数组

        支持两种常见的ONNX模型输出格式：
        1. 直接输出sentence_embedding (key: "sentence_embedding")
        2. 输出last_hidden_state，需要mean pooling (key: "last_hidden_state")
        """
        if self.tokenizer is None or self.session is None:
            raise RuntimeError("模型未加载")

        # Tokenize
        encoding = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoding], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoding], dtype=np.int64)

        # 获取输入名称
        input_names = [inp.name for inp in self.session.get_inputs()]
        feed_dict = {}

        # 根据模型输入构建feed字典
        if "input_ids" in input_names:
            feed_dict["input_ids"] = input_ids
        if "attention_mask" in input_names:
            feed_dict["attention_mask"] = attention_mask

        # 某些模型可能需要token_type_ids
        if "token_type_ids" in input_names:
            feed_dict["token_type_ids"] = np.zeros_like(input_ids)

        # 运行推理
        outputs = self.session.run(None, feed_dict)
        output_names = [out.name for out in self.session.get_outputs()]

        # 处理输出
        if "sentence_embedding" in output_names:
            # 直接返回sentence embedding
            idx = output_names.index("sentence_embedding")
            embeddings = outputs[idx]
        elif "last_hidden_state" in output_names:
            # 需要mean pooling
            idx = output_names.index("last_hidden_state")
            last_hidden_state = outputs[idx]
            embeddings = _mean_pooling(last_hidden_state, attention_mask)
            # L2归一化
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            # 默认返回第一个输出
            embeddings = outputs[0]
            # 如果是3D (batch, seq_len, hidden_dim)，进行mean pooling
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
        async with self._model_lock:
            if self.session is None:
                logger.info("[ONNXEmbedding] 模型未加载，无需卸载")
                return True

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._cleanup_resources)

    def force_unload_sync(self) -> bool:
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
                "ONNXEmbedding_tokenizer_path": "",
                "ONNXEmbedding_optimization_level": "disable",
                "ONNXEmbedding_max_length": 256,
                "provider_type": "embedding",
                "enable": True,
                "embedding_dimensions": 384,
            }
        except KeyError:
            logger.error("[ONNXEmbedding] AstrBot 配置结构异常，无法注册 Provider")
            return False

        try:
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "ONNXEmbedding_path"
            ] = {
                "description": "ONNX 模型路径（目录或.onnx文件）",
                "type": "string",
            }
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "ONNXEmbedding_tokenizer_path"
            ] = {
                "description": "Tokenizer 文件路径（可选，默认从模型目录查找）",
                "type": "string",
            }
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "ONNXEmbedding_optimization_level"
            ] = {
                "description": "ONNX Runtime 图优化级别（disable/basic/extended/all），默认disable避免兼容性问题",
                "type": "string",
            }
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                "ONNXEmbedding_max_length"
            ] = {
                "description": "最大序列长度（默认256），超过会被截断",
                "type": "int",
            }
        except KeyError:
            logger.error("[ONNXEmbedding] AstrBot 配置结构异常，无法注册 Provider")
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                "config_template"
            ].pop("ONNXEmbedding", None)
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

        # ---- Chat Provider 注册 ----
        chat_already_registered = False
        if isinstance(provider_registry, list):
            for p in provider_registry:
                if getattr(p, "type", None) == "ONNXChat":
                    chat_already_registered = True
                    break

        if not chat_already_registered:
            register_ONNXChatProvider()
            # 注册 Chat Provider 配置
            try:
                CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
                    "config_template"
                ]["ONNXChat"] = {
                    "id": "ONNXChat",
                    "type": "ONNXChat",
                    "provider": "Local",
                    "ONNXChat_model_path": "",
                    "ONNXChat_tokenizer_path": "",
                    "ONNXChat_model_name": "onnx-chat-model",
                    "ONNXChat_max_length": 512,
                    "ONNXChat_max_new_tokens": 128,
                    "ONNXChat_temperature": 0.8,
                    "ONNXChat_top_p": 0.9,
                    "ONNXChat_top_k": 50,
                    "provider_type": "chat_completion",
                    "enable": True,
                }
                CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                    "ONNXChat_model_path"
                ] = {
                    "description": "ONNX 聊天模型路径（目录或.onnx文件）",
                    "type": "string",
                }
                CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                    "ONNXChat_max_new_tokens"
                ] = {
                    "description": "最大生成token数（默认128）",
                    "type": "int",
                }
                CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"][
                    "ONNXChat_temperature"
                ] = {
                    "description": "采样温度（默认0.8）",
                    "type": "float",
                }
            except KeyError:
                logger.warning("[ONNXChat] 无法注册 Chat Provider 配置")

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
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "ONNXEmbedding_tokenizer_path", None
            )
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "ONNXEmbedding_optimization_level", None
            )
            CONFIG_METADATA_2["provider_group"]["metadata"]["provider"]["items"].pop(
                "ONNXEmbedding_max_length", None
            )
        except KeyError:
            pass

        self._registered = False
        logger.info("[ONNXEmbedding] 配置已清理")

    # --------------------------------------------------------
    # Commands
    # --------------------------------------------------------
    @filter.command_group("onnx")
    def onnx(self):
        pass

    @onnx.command("help")
    async def help_cmd(self, event: AstrMessageEvent):
        """获取帮助信息"""
        help_text = [
            "ONNXEmbedding 插件 - 基于ONNX Runtime的轻量级嵌入向量生成插件",
            "/onnx register                      注册 Provider",
            "/onnx redb                          重新加载数据库",
            "/onnx kbinfo                        获取所有数据库以及其对应的embedding_provider_id",
            "/onnx unload [embedding_provider_id] 卸载指定Provider的权重",
        ]
        yield event.plain_result("\n".join(help_text))

    @onnx.command("redb")
    async def redb(self, event: AstrMessageEvent):
        """重新加载数据库，防止在astrbot初始化后出现ONNXEmbeddingProvider未注册数据库加载失败的情况"""
        await self.context.kb_manager.load_kbs()
        yield event.plain_result("[ONNXEmbedding] 数据库已重新加载")

    @onnx.command("register")
    async def register_cmd(self, event: AstrMessageEvent):
        """主动将ONNXEmbedding注册到嵌入式向量提供商"""
        yield event.plain_result("[ONNXEmbedding] 正在注册 Provider")
        if self._register_config():
            yield event.plain_result("[ONNXEmbedding] 注册 Provider 成功")
        await self.context.kb_manager.load_kbs()

    @onnx.command("kbinfo")
    async def get_kb_name_epid(self, event: AstrMessageEvent):
        """获取所有数据库以及其对应的编码器"""
        outputtext = []
        for kb_helper in self.context.kb_manager.kb_insts.values():
            outputtext.append(
                f"数据库名称:{kb_helper.kb.kb_name}, 编码器:{kb_helper.kb.embedding_provider_id}"
            )
        yield event.plain_result(f"可用数据库:\n" + "\n".join(outputtext))
        logger.info(f"[ONNXEmbedding] 可用数据库:\n" + "\n".join(outputtext))

    @onnx.command("unload")
    async def unload_kbw(self, event: AstrMessageEvent, embedding_provider_id: str):
        """清理权重，防止用不到权重时占用太多内存"""
        pm = self.context.provider_manager.get_provider_by_id(embedding_provider_id)
        if isinstance(pm, ONNXEmbeddingProvider):
            yield event.plain_result(f"[ONNXEmbedding] 正在清理权重")
            logger.info(f"[ONNXEmbedding] 正在清理权重")
            await pm.unload_model()
            yield event.plain_result(f"[ONNXEmbedding] 清理权重成功")
            logger.info(f"[ONNXEmbedding] 清理权重成功")
        else:
            yield event.plain_result(
                f"[ONNXEmbedding] 编码器实例:{embedding_provider_id},不为ONNXEmbeddingProvider"
            )
            logger.info(
                f"[ONNXEmbedding] 编码器实例:{embedding_provider_id},不为ONNXEmbeddingProvider"
            )

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
