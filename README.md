# ONNXEmbedding - AstrBot ONNX / OpenVINO Embedding、Rerank、Chat 插件

一个为 [AstrBot](https://github.com/AstrBotDevs/astrbot) 设计的本地推理插件，支持 **Embedding / Rerank / Chat** 三类 Provider，并兼容 **ONNX Runtime** 与 **OpenVINO GenAI**。相较于传统 PyTorch 方案，它更轻量、部署门槛更低，也更适合本地知识库与轻量聊天模型场景。

## 功能特性

- 🚀 **多后端支持**：支持 ONNX Runtime、ONNX Runtime GenAI、OpenVINO、OpenVINO GenAI
- 💬 **本地 Chat Provider**：支持加载 ONNX / OpenVINO 本地聊天模型（如 Qwen、Phi 等）
- 📦 **Embedding 向量生成**：支持 ONNX Embedding 模型
- 🔄 **Rerank 重排序**：内置 ONNX Rerank Provider，支持知识库检索重排
- 🔁 **重启恢复加载**：修复 AstrBot 重启后 ONNX Provider 可能丢失的问题
- 📥 **自动下载模型**：支持通过插件命令与配置自动从 Hugging Face 下载模型
- 🧠 **思考内容分离**：自动将 Qwen 等模型输出中的 `<think>...</think>` 提取为 AstrBot reasoning 内容，避免与正文混杂
- 🌐 **HuggingFace 镜像支持**：支持配置镜像地址加速模型下载
- ⏱️ **自动卸载**：支持长时间未使用模型自动释放内存

## 安装要求

### 系统依赖

- Python 3.10+
- AstrBot 框架

### Python 依赖

基础依赖（必需）：

```bash
pip install onnxruntime tokenizers numpy huggingface_hub
```

GPU 加速（可选）：

```bash
# NVIDIA CUDA
pip install onnxruntime-gpu

# DirectML (Windows)
pip install onnxruntime-directml
```

Chat / OpenVINO 后端（可选）：

```bash
# OpenVINO GenAI（推荐 Intel CPU / GPU）
pip install openvino openvino-genai

# ONNX Runtime GenAI（普通 ONNX 对话模型）
pip install onnxruntime-genai
```

> 对于 `OpenVINO/Qwen3-0.6B-fp16-ov` 一类模型，推荐直接使用 `openvino` 或 `auto` 后端。

## 配置说明

### 插件级配置

在插件配置页面可以设置：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `auto_start` | int | 0 | 是否开机自动注册 Provider |
| `huggingface_mirror` | string | "" | HuggingFace 镜像地址（如：https://hf-mirror.com） |
| `auto_download` | int | 1 | 模型不存在时是否自动下载 |
| `auto_unload_timeout` | int | 0 | 模型自动卸载超时时间（分钟，0 表示禁用） |

### Provider 配置

创建 Embedding Provider 时：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `ONNXEmbedding_path` | `sentence-transformers/all-MiniLM-L6-v2` | 模型路径，支持 HuggingFace 模型名或本地路径 |
| `embedding_dimensions` | 384 | 嵌入向量维度 |

创建 Rerank Provider 时：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `ONNXRerank_path` | `BAAI/bge-reranker-base` | Rerank 模型路径 |
| `ONNXRerank_max_length` | 512 | 最大序列长度 |

创建 Chat Provider 时：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `ONNXChat_path` | `microsoft/phi-2` | 聊天模型路径，支持 HuggingFace 模型名或本地目录 |
| `ONNXChat_backend` | `auto` | `auto` / `openvino` / `onnxruntime` |
| `ONNXChat_device` | `CPU` | OpenVINO 设备，如 `CPU` / `GPU` / `AUTO` / `NPU` |
| `ONNXChat_max_new_tokens` | 512 | 最大生成 token 数 |
| `ONNXChat_temperature` | 0.7 | 生成温度 |
| `ONNXChat_context_length` | 2048 | 上下文长度提示 |

> 若在 AstrBot 中开启 `display_reasoning_text`，Qwen 等模型输出的 `<think>` 内容会被自动展示为“思考内容”，正文中不会再混入这些标签。

## 使用方法

### 1. 创建 Embedding Provider

1. 在 AstrBot 管理面板 → Provider → 嵌入(Embedding) → 新增
2. 选择类型 `ONNXEmbedding`
3. 配置模型路径（如 `sentence-transformers/all-MiniLM-L6-v2`）
4. 设置嵌入维度（根据模型调整）

### 2. 创建 Rerank Provider

1. 在知识库设置中选择 Rerank Provider
2. 选择类型 `ONNXRerank`
3. 配置模型路径（推荐 `BAAI/bge-reranker-base`）

### 3. 创建 Chat Provider

1. 在 AstrBot 管理面板 → Provider → 对话(Chat Completion) → 新增
2. 选择类型 `ONNXChatProvider`
3. 根据模型格式设置：
   - OpenVINO 模型：`ONNXChat_backend = openvino` 或 `auto`
   - 普通 ONNX 模型：`ONNXChat_backend = onnxruntime` 或 `auto`
4. 例如：
   - `OpenVINO/Qwen3-0.6B-fp16-ov`
   - `onnx-community/Qwen3.5-0.8B-ONNX`

### 4. 管理命令

```bash
/onnx <知识库名> <查询内容>
/onnx_dl <embed|rerank|chat> <HuggingFace模型名>
/onnx_info
```

示例：

```bash
/onnx 我的知识库 如何配置插件
/onnx_dl chat OpenVINO/Qwen3-0.6B-fp16-ov
/onnx_info
```

其中：
- `/onnx`：直接查询知识库
- `/onnx_dl`：下载模型到插件数据目录
- `/onnx_info`：查看当前 ONNX Provider 与已下载模型状态

## 后端选择

插件支持两种推理后端：

### ONNX Runtime（默认）

- 通用性好，支持多种硬件
- 自动检测可用的 Execution Provider（CUDA、DirectML、CPU 等）

### OpenVINO

- Intel 硬件优化，特别是 Intel iGPU
- 安装 OpenVINO 后自动启用
- 自动检测可用设备（GPU/CPU）

当 OpenVINO 已安装时，插件会优先尝试使用 OpenVINO 后端；如果失败则回退到 ONNX Runtime。

## 推荐模型

### Embedding 模型

| 模型 | 语言 | 维度 | 大小 | 说明 |
|------|------|------|------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 英语 | 384 | ~80MB | 推荐，快速高效 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 多语言 | 384 | ~120MB | 多语言支持 |
| `BAAI/bge-small-zh-v1.5` | 中文 | 512 | ~100MB | 中文优化 |

### Rerank 模型

| 模型 | 语言 | 大小 | 说明 |
|------|------|------|------|
| `BAAI/bge-reranker-base` | 多语言 | ~1GB | 推荐，效果好 |
| `BAAI/bge-reranker-large` | 多语言 | ~3GB | 更高精度 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 英语 | ~80MB | 英文专用 |

## HuggingFace 镜像配置

国内用户可配置镜像加速下载：

```json
{
  "huggingface_mirror": "https://hf-mirror.com"
}
```

常用镜像：
- https://hf-mirror.com
- https://huggingface.co（官方源）

## 故障排除

### 1. 缺少依赖

```bash
# 基础依赖
pip install onnxruntime tokenizers numpy

# OpenVINO（可选）
pip install openvino
```

### 2. Intel GPU 加速

如果使用 Intel iGPU，推荐安装 OpenVINO：

```bash
pip install openvino
```

### 3. 模型下载失败

- 检查网络连接
- 配置 HuggingFace 镜像地址
- 手动下载模型文件

### 4. 内存不足

- 设置 `auto_unload_timeout` 自动卸载长时间未使用的模型
- 选择更小的模型

## 版本历史

### v2.2.0

- 新增对 Qwen 等模型 `<think>...</think>` 的自动解析，思考内容会单独进入 AstrBot reasoning 展示
- 改进 ONNX / OpenVINO Chat 模型加载逻辑，兼容缺少 `genai_config.json` 的 Hugging Face 模型仓库
- 优化 Hugging Face 自动下载，优先使用 `snapshot_download` 并在文件损坏时自动重试修复
- 改进 OpenVINO Chat 加载流程，按 Hugging Face / OpenVINO GenAI 文档启用 tokenizer chat template

### v2.1.0

- 新增本地 `ONNXChatProvider`
- 修复 AstrBot 重启后 ONNX Provider 可能丢失的问题
- 新增 `/onnx_dl` 与 `/onnx_info` 命令
- 新增 OpenVINO / ORT GenAI 双后端支持

### v1.0.0

- 初始版本发布
- 支持 ONNX Runtime 推理
- 基本的嵌入向量生成功能

## 致谢

- [ONNX Runtime](https://onnxruntime.ai/) - 高性能推理引擎
- [OpenVINO](https://docs.openvino.ai/) - Intel 优化推理引擎
- [Hugging Face](https://huggingface.co/) - 模型托管平台
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 插件框架