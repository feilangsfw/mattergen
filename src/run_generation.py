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
# 离线模式需要手动设置环境变量，如需要离线运行，可以使用:
# os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

# 获取当前脚本的目录，并设置项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 添加项目根目录到sys.path，以便导入mattergen
sys.path.insert(0, str(PROJECT_ROOT))

from mattergen.scripts.generate import main
from mattergen.common.data.types import TargetProperty

# 定义参数
MODEL_NAME = "mp20_csp_260211"  # 使用预训练模型名，从缓存加载
# 本地模型路径，基于MODEL_NAME构建
LOCAL_MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / MODEL_NAME)
# 本地模型路径，直接使用下载缓存
# MODEL_PATH = "/home/wczhou/data_linked/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/f981d511721156a4949caae7dcac6979afe97f3a/checkpoints/mattergen_base"
RESULTS_PATH = str(PROJECT_ROOT / "results")  # 项目根目录下的results目录
BATCH_SIZE = 16 # 每次迭代生成的样本数量
NUM_BATCHES = 2 # 生成次数
CHECKPOINT_EPOCH = "last" # 从模型检查点中加载哪个epoch的权重

# 可选参数，根据需要取消注释
COMPOSITIONS = [      # 只支持CSP模型
    {"Y": 1, "Ba": 2, "Cu":3,"O": 7}
]
""" PROPERTIES_TO_CONDITION_ON: TargetProperty = {
    #"space_group": 225,
    "chemical_system": "Y-Ba-Cu-O"} """
DIFFUSION_GUIDANCE_FACTOR = 2.0

def run_generation():
    # 检查模型目录是否存在
    model_dir = Path(LOCAL_MODEL_PATH)
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

    print(f"使用镜像端点: {os.environ.get('HF_ENDPOINT', '未设置')}")
    print(f"缓存目录: {os.environ.get('HF_HOME', '未设置')}")
    print(f"离线模式: {'启用' if os.environ.get('HF_HUB_OFFLINE', '0') == '1' else '禁用'}")
    print(f"模型路径: {LOCAL_MODEL_PATH}")

    # 调用生成函数，使用预训练模型名
    structures = main(
        output_path=RESULTS_PATH,
        # pretrained_name=MODEL_NAME,  # 使用预训练模型名，会自动使用缓存
        model_path=LOCAL_MODEL_PATH,  # 使用本地模型路径
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        checkpoint_epoch=CHECKPOINT_EPOCH,
        target_compositions=COMPOSITIONS, # 只支持CSP模型
        #properties_to_condition_on=PROPERTIES_TO_CONDITION_ON,
        diffusion_guidance_factor=DIFFUSION_GUIDANCE_FACTOR,
        sampling_config_name="csp",  # 添加这一行，指定使用CSP采样配置
    )

    print(f"生成完成，共生成 {len(structures)} 个结构")
    print(f"结果保存在: {results_dir.absolute()}")


if __name__ == "__main__":
    run_generation()