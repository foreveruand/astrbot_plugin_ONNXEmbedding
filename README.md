# ONNXEmbedding - AstrBot ONNX 嵌入向量生成插件

一个为 [AstrBot](https://github.com/AstrBotDevs/astrbot) 框架设计的 **ONNX Runtime** 嵌入向量生成插件，相比传统的 Sentence Transformers 方案具有更低的配置要求和更快的推理速度。

## 功能特性

- 🚀 **ONNX Runtime 加速**: 使用 ONNX Runtime 推理引擎，无需 PyTorch，配置要求更低
- 📦 **轻量级部署**: 相比 PyTorch 版本，内存占用和磁盘空间需求大幅减少
- 🔧 **无缝集成**: 作为 AstrBot 的 Provider 适配器，可直接在框架配置中使用
- 📦 **自动配置注册**: 插件启动时自动注册配置项到 AstrBot 全局配置
- 🧹 **资源清理**: 插件卸载时自动清理注册的配置和适配器
- 🔌 **即插即用**: 简单的安装和配置流程

## 安装要求

### 系统依赖

- Python 3.10+
- AstrBot 框架

### Python 依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install onnxruntime tokenizers numpy
```

插件会自动检查依赖，如果缺少必要的库，会在初始化时提示安装。

## 配置说明

### 自动注册的配置项

插件初始化时会自动向 AstrBot 注册以下配置：

```json
{
  "provider_group": {
    "metadata": {
      "provider": {
        "config_template": {
          "ONNXEmbedding": {
            "id": "ONNXEmbedding",
            "type": "ONNXEmbedding",
            "provider": "Local",
            "ONNXEmbedding_path": "./all-MiniLM-L6-v2/",
            "ONNXEmbedding_tokenizer_path": "",
            "provider_type": "embedding",
            "enable": true,
            "embedding_dimensions": 384
          }
        },
        "items": {
          "ONNXEmbedding_path": {
            "description": "ONNX 模型路径（目录或.onnx文件）",
            "type": "string"
          },
          "ONNXEmbedding_tokenizer_path": {
            "description": "Tokenizer 文件路径（可选，默认从模型目录查找）",
            "type": "string"
          }
        }
      }
    }
  }
}
```

### 配置参数详解

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ONNXEmbedding_path` | string | `"all-MiniLM-L6-v2"` | ONNX 模型路径，支持相对路径和绝对路径，可以是目录或.onnx文件 |
| `ONNXEmbedding_tokenizer_path` | string | `""` | Tokenizer 文件路径，可选，默认从模型目录自动查找 |
| `embedding_dimensions` | integer | `384` | 嵌入向量的维度 |
| `enable` | boolean | `true` | 是否启用该 provider |
| `provider_type` | string | `"embedding"` | Provider 类型 |

### 路径说明

- **相对路径**: 相对于 AstrBot 的 `data_dir` 目录
- **绝对路径**: 直接使用指定的完整路径

## 使用方法

### 1. 作为嵌入向量提供者

可以在插件配置里面选择开机自启，第一次需要用户手动操作启动。使用指令 `/onnx register` 将提供商注册到嵌入向量提供者，然后可以通过嵌入式模型的创建页面创建这个嵌入向量提供者。

### 2. 插件命令

使用 `/onnx help` 获取帮助：

- `/onnx register` - 注册 Provider
- `/onnx redb` - 重新加载数据库
- `/onnx kbinfo` - 获取所有数据库以及其对应的 embedding_provider_id
- `/onnx unload [embedding_provider_id]` - 卸载指定 Provider 的权重

## 模型支持

### 预训练模型

插件默认使用 `all-MiniLM-L6-v2` 模型（384维）。

### 如何获取 ONNX 模型

1. **从 Hugging Face 下载转换后的模型**: 许多 Sentence Transformer 模型已经有社区转换的 ONNX 版本
   
2. **手动转换**: 使用 `optimum` 库将 PyTorch 模型转换为 ONNX 格式：
   ```bash
   pip install optimum[exporters]
   optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./onnx-model/
   ```

3. **使用预转换模型**: 推荐从 [Hugging Face](https://huggingface.co/models?library=onnx) 下载已转换的 ONNX 模型

### 模型目录结构

ONNX 模型目录应包含以下文件：

```
model_dir/
├── model.onnx          # ONNX 模型文件
└── tokenizer.json      # Tokenizer 配置文件
```

### 推荐模型

- `sentence-transformers/all-MiniLM-L6-v2` (推荐，英语，384维)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (多语言，384维)

## 性能对比

| 特性 | Sentence Transformers (PyTorch) | ONNX Runtime |
|------|----------------------------------|--------------|
| 依赖大小 | ~500MB+ | ~100MB |
| 内存占用 | 较高 | 较低 |
| 推理速度 | 快 | 更快 |
| GPU 支持 | CUDA | CUDA/DirectML/ROCm |
| 配置复杂度 | 高 | 低 |

## 开发说明

### 生命周期方法

- `initialize()`: 插件启动时调用，注册配置和适配器
- `terminate()`: 插件停止时调用，清理配置和适配器

### 日志

插件使用 AstrBot 的日志系统，日志前缀为 `[ONNXEmbedding]`。

## 故障排除

### 常见问题

#### 1. 导入错误：缺少 onnxruntime

```bash
# 安装依赖
pip install onnxruntime
```

对于 GPU 支持：
```bash
# CUDA
pip install onnxruntime-gpu

# DirectML (Windows)
pip install onnxruntime-directml
```

#### 2. 模型加载失败

- 检查模型路径是否正确
- 确认 model.onnx 文件存在且完整
- 检查 tokenizer.json 是否存在

#### 3. 配置未注册

- 确认插件已正确加载
- 检查插件初始化日志
- 重启 AstrBot

#### 4. Tokenizer 未找到

- 确保模型目录包含 `tokenizer.json` 文件
- 或在配置中显式指定 `ONNXEmbedding_tokenizer_path`

## 版本历史

### v1.0.0

- 初始版本发布
- 支持 ONNX Runtime 推理
- 自动配置注册和清理
- 提供基本的嵌入向量生成功能

## 致谢

- [ONNX Runtime](https://onnxruntime.ai/) - 用于高性能推理
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) - 用于文本分词
- [Sentence Transformers](https://www.sbert.net/) - 提供原始模型架构
- [AstrBot](https://github.com/AstrBotDevs/AstrBot) - 提供插件框架
