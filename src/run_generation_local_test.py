#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行MatterGen生成任务的便捷脚本
"""

import os
from pathlib import Path
from mattergen.scripts.generate import main

# 在脚本开始时设置环境变量
os.environ["HF_HOME"] = "/home/wczhou/data_linked/.cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制使用本地缓存

# 获取当前脚本的目录，并设置项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 定义参数 - 使用本地模型路径，指向包含config.yaml的目录
MODEL_PATH = "/home/wczhou/data_linked/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/f981d511721156a4949caae7dcac6979afe97f3a/checkpoints/mattergen_base"
RESULTS_PATH = str(PROJECT_ROOT / "results")  # 项目根目录下的results目录
BATCH_SIZE = 16
NUM_BATCHES = 1
CHECKPOINT_EPOCH = "last"
# 可选参数，根据需要取消注释
# PROPERTIES_TO_CONDITION_ON = {"energy_above_hull": 0.05, "chemical_system": "Li-O"}
# DIFFUSION_GUIDANCE_FACTOR = 2.0

def run_generation():
    # 检查模型目录是否存在
    model_dir = Path(MODEL_PATH)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir.absolute()}")
    
    # 检查config.yaml是否存在
    config_file = model_dir / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file.absolute()}")
    
    # 检查输出目录是否存在，不存在则报错
    results_dir = Path(RESULTS_PATH)
    if not results_dir.exists():
        raise FileNotFoundError(f"输出目录不存在: {results_dir.absolute()}")
    
    # 调用生成函数，使用本地模型路径
    structures = main(
        output_path=RESULTS_PATH,
        model_path=MODEL_PATH,  # 使用本地模型路径
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