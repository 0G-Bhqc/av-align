"""后端适配器（L2）。只转发布局计划，禁止携带自己的对齐逻辑（方案 4.1）。

- transformers_adapter：后端 A（Transformers / Qwen2.5-Omni），CPU 级只读重述；
- mlx_adapter：后端 B（mlx-vlm / qwen3_omni_moe），代码就绪，Metal 执行待 Apple Silicon。
"""
