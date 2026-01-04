#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行MatterGen生成任务的便捷脚本
"""

import os
import sys
# 在任何其他import之前设置环境变量
os.environ["HF_HOME"] = "/home/wczhou/data_linked/.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 移除离线模式，允许在线下载
# os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

# 获取当前脚本的目录，并设置项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 添加项目根目录到sys.path，以便导入mattergen
sys.path.insert(0, str(PROJECT_ROOT))

from mattergen.scripts.generate import main

# 定义参数
MODEL_NAME = "mp_20_base"  # 使用预训练模型名，从缓存加载
LOCAL_MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / MODEL_NAME)  # 本地模型路径，基于MODEL_NAME构建
RESULTS_PATH = str(PROJECT_ROOT / "results")  # 项目根目录下的results目录
BATCH_SIZE = 16
NUM_BATCHES = 1
CHECKPOINT_EPOCH = "last"
# 可选参数，根据需要取消注释
# PROPERTIES_TO_CONDITION_ON = {"energy_above_hull": 0.05, "chemical_system": "Li-O"}
# DIFFUSION_GUIDANCE_FACTOR = 2.0

def run_generation():
    # 检查输出目录是否存在，不存在则报错
    results_dir = Path(RESULTS_PATH)
    if not results_dir.exists():
        raise FileNotFoundError(f"输出目录不存在: {results_dir.absolute()}")

    print(f"使用镜像端点: {os.environ.get('HF_ENDPOINT', '未设置')}")
    print(f"缓存目录: {os.environ.get('HF_HOME', '未设置')}")

    # 调用生成函数，使用预训练模型名
    structures = main(
        output_path=RESULTS_PATH,
        pretrained_name=MODEL_NAME,  # 使用预训练模型名，会自动使用缓存
        # model_path=LOCAL_MODEL_PATH,  # 使用本地模型路径
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        checkpoint_epoch=CHECKPOINT_EPOCH,
        # properties_to_condition_on=PROPERTIES_TO_CONDITION_ON,
        # diffusion_guidance_factor=DIFFUSION_GUIDANCE_FACTOR,
    )

    print(f"生成完成，共生成 {len(structures)} 个结构")
    print(f"结果保存在: {results_dir.absolute()}")


if __name__ == "__main__":
    run_generation()