# ONNXEmbedding - AstrBot ONNX 嵌入向量生成插件

一个为 [AstrBot](https://github.com/AstrBotDevs/astrbot) 框架设计的 **ONNX Runtime / OpenVINO** 嵌入向量生成和重排序插件，相比传统的 PyTorch 方案具有更低的配置要求和更快的推理速度。

## 功能特性

- 🚀 **多后端支持**: 支持 ONNX Runtime 和 OpenVINO 两种推理后端
- 🖥️ **硬件加速**: 支持 CPU、NVIDIA GPU (CUDA)、AMD GPU (ROCm)、Intel GPU (OpenVINO)、DirectML
- 📦 **嵌入向量生成**: 支持 ONNX Embedding 模型
- 🔄 **重排序支持**: 内置 ONNX Rerank Provider，支持知识库检索重排序
- 📦 **轻量级部署**: 相比 PyTorch 版本，内存占用和磁盘空间需求大幅减少
- 🔧 **无缝集成**: 作为 AstrBot 的 Provider 适配器，可直接在框架配置中使用
- 🌐 **HuggingFace 镜像**: 支持配置 HuggingFace 镜像地址，加速模型下载
- ⏱️ **自动卸载**: 支持配置模型自动卸载超时时间，节省内存

## 安装要求

### 系统依赖

- Python 3.10+
- AstrBot 框架

### Python 依赖

基础依赖（必需）：

```bash
pip install onnxruntime tokenizers numpy
```

GPU 加速（可选）：

```bash
# NVIDIA CUDA
pip install onnxruntime-gpu

# DirectML (Windows)
pip install onnxruntime-directml
```

OpenVINO 后端（可选，推荐 Intel GPU 用户使用）：

```bash
pip install openvino
```

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

## 使用方法

### 1. 创建 Embedding Provider

1. 在 AstrBot 管理面板 → Provider → 嵌入(Embedding) → 新增
2. 选择类型 `ONNXEmbedding`
3. 配置模型路径（支持 HuggingFace 模型名，如 `sentence-transformers/all-MiniLM-L6-v2`）
4. 设置嵌入维度（根据模型调整）

### 2. 创建 Rerank Provider

1. 在知识库设置中选择 Rerank Provider
2. 选择类型 `ONNXRerank`
3. 配置模型路径（推荐 `BAAI/bge-reranker-base`）

### 3. 直接查询命令

```
/onnx <知识库名> <查询内容>
```

示例：
```
/onnx 我的知识库 如何配置插件
```

返回知识库中最相关的 2 条结果。

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

### v2.0.0

- 新增 OpenVINO 后端支持，优化 Intel GPU 性能
- 新增 ONNX Rerank Provider
- 新增 `/onnx` 命令直接查询知识库
- 新增 HuggingFace 镜像配置
- 新增模型自动卸载功能
- 简化配置项，移除冗余参数
- 移除 Chat Provider（建议使用专门的对话模型）

### v1.0.0

- 初始版本发布
- 支持 ONNX Runtime 推理
- 基本的嵌入向量生成功能

## 致谢

- [ONNX Runtime](https://onnxruntime.ai/) - 高性能推理引擎
- [OpenVINO](https://docs.openvino.ai/) - Intel 优化推理引擎
- [Hugging Face](https://huggingface.co/) - 模型托管平台
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 插件框架