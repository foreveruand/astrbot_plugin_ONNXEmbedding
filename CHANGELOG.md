# Changelog

## v2.2.0

- Added Qwen-style `<think>...</think>` parsing for `ONNXChatProvider`; reasoning content is now separated from the visible answer and passed to AstrBot's reasoning display.
- Improved ONNX/OpenVINO chat-model compatibility for Hugging Face repos that do not ship `genai_config.json`.
- Aligned OpenVINO loading and downloading with the Hugging Face / OpenVINO GenAI guidance.
- Improved auto-download reliability and repair handling for incomplete or corrupted model files.

## v2.1.0

- Added `ONNXChatProvider` for local ONNX / OpenVINO chat models.
- Added provider recovery after AstrBot restart.
- Added `/onnx_dl` and `/onnx_info` management commands.

## v1.0.0

- Initial release with ONNX embedding support.
