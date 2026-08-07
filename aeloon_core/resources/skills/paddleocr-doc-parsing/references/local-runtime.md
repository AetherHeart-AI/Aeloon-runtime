# 本地运行约束

- 使用 `paddlepaddle` 与 `paddleocr[doc-parser]` 的独立 Python 环境。
- 通过 `PADDLE_PDX_CACHE_HOME` 将模型缓存固定到 Aeloon 数据目录。
- `--offline` 设置 Hugging Face 与 Transformers 的离线标志，并在解析期间阻断 socket 连接；只有缓存完整时才能成功。
- CPU 是默认设备。GPU 或其他推理引擎必须由用户环境明确提供，不能自动安装驱动。
- Apple Silicon 可采用 PaddleOCR 官方的手动本地安装路径；其他 ARM64 平台必须在发布前单独验证，不得把未经验证的平台写成已支持。
- 本技能不读取访问令牌、不实例化 HTTP 客户端，也不调用 `paddleocr api`。
