# 本地运行约束

- 使用 Aeloon 安装包内置的 `paddlepaddle` 与 `paddleocr[doc-parser]` 运行时。
- 通过 `PADDLE_PDX_CACHE_HOME` 将模型缓存固定到 Aeloon 数据目录。
- `--offline` 设置 Hugging Face 与 Transformers 的离线标志，并在解析期间阻断 socket 连接；只有缓存完整时才能成功。
- CPU 是默认设备。GPU 或其他推理引擎必须由用户环境明确提供，不能自动安装驱动。
- Apple Silicon 与 Ubuntu ARM64 必须分别通过官方 Aeloon 发布包的独立构建和运行时自检；其他平台不得写成已支持。
- 本技能不读取访问令牌、不实例化 HTTP 客户端，也不调用 `paddleocr api`。
