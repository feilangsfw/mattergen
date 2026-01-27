#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行MatterGen训练任务的便捷脚本
此脚本等价于运行:
mattergen-train data_module=mp_20 ~trainer.logger
但允许通过Python代码直接修改参数，而不需要修改YAML文件
"""

import os
import sys
from pathlib import Path

# 在任何其他import之前设置环境变量，使用Path对象然后转换为字符串
PROJECT_ROOT = Path(__file__).parent.parent  # /home/wczhou/data_linked/projects/mattergen

# 关键：在导入mattergen模块前设置环境变量
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["MATTERGEN_MODELS_PROJECT_ROOT"] = str(PROJECT_ROOT)

# 重要：预设PYTHONPATH，确保Python能找到正确的mattergen模块
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf


# 现在导入mattergen相关模块
from mattergen.diffusion.config import Config
from mattergen.diffusion.run import main

def run_training():
    # 设置Tensor Core加速
    torch.set_float32_matmul_precision("high")

    # 配置参数 - 这些是你可以根据需要修改的参数
    # 这些参数等价于在命令行中使用 +param=value 的形式覆盖YAML配置
    DATA_MODULE_NAME = "mp_20"  # 等价于命令行: data_module=mp_20
    DATASET_PATH = str(PROJECT_ROOT / "datasets" / "cache" / DATA_MODULE_NAME)  # 数据集路径

    # 数据模块参数 (来自 conf/data_module/mp_20.yaml)
    BATCH_SIZE_TRAIN = 32  # 训练批处理大小，降低以节省内存
    BATCH_SIZE_VAL = 16     # 验证批处理大小，降低以节省内存
    BATCH_SIZE_TEST = 16    # 测试批处理大小，降低以节省内存

    # 训练器参数 (来自 conf/trainer/default.yaml)
    ACCELERATOR = "gpu"     # 使用GPU
    DEVICES = "auto"        # 自动检测可用GPU
    PRECISION = 32          # 精度 (32表示FP32)
    ACCUMULATE_GRAD_BATCHES = 4  # 梯度累积步数，补偿较小的批处理大小

    # 优化器参数 (来自 conf/lightning_module/default.yaml)
    LEARNING_RATE = 0.0001  # 学习率

    # 可选的高级参数（可根据需要取消注释并调整）
    # MAX_EPOCHS = 900        # 最大训练轮数
    # GRADIENT_CLIP_VAL = 0.5 # 梯度裁剪值
    # CHECK_VAL_EVERY_N_EPOCH = 5  # 每N个epoch验证一次
    # SCHEDULER_FACTOR = 0.6  # 学习率衰减因子
    # SCHEDULER_PATIENCE = 100 # 调度器耐心值
    # SCHEDULER_MIN_LR = 1e-6   # 最小学习率
    # HIDDEN_DIM = 256        # 隐藏层维度，降低以节省内存
    # GEMNET_NUM_BLOCKS = 2   # GemNet块的数量，降低以节省内存
    # GEMNET_CUTOFF = 7.0     # GemNet截断距离

    print("="*80)
    print("路径变量信息:")
    print(f"PROJECT_ROOT (Path object): {PROJECT_ROOT}")
    print(f"Environment PROJECT_ROOT: {os.environ.get('PROJECT_ROOT', 'NOT SET')}")
    print(f"Environment MATTERGEN_MODELS_PROJECT_ROOT: {os.environ.get('MATTERGEN_MODELS_PROJECT_ROOT', 'NOT SET')}")
    print("="*80)
    print("此脚本等価于运行以下命令:")
    print(f"mattergen-train data_module={DATA_MODULE_NAME} ~trainer.logger")
    print("+data_module.root_dir='{}'".format(DATASET_PATH))
    print(f"+data_module.batch_size.train={BATCH_SIZE_TRAIN}")
    print(f"+data_module.batch_size.val={BATCH_SIZE_VAL}")
    print(f"+data_module.batch_size.test={BATCH_SIZE_TEST}")
    print(f"+trainer.accelerator={ACCELERATOR}")
    print(f"+trainer.devices={DEVICES}")
    print(f"+trainer.precision={PRECISION}")
    print(f"+trainer.accumulate_grad_batches={ACCUMULATE_GRAD_BATCHES}")
    print(f"+lightning_module.optimizer_partial.lr={LEARNING_RATE}")
    # 如需使用高级参数，请取消下面的注释并调整参数值
    # print(f"+trainer.max_epochs={MAX_EPOCHS}")
    # print(f"+trainer.gradient_clip_val={GRADIENT_CLIP_VAL}")
    # print(f"+trainer.check_val_every_n_epoch={CHECK_VAL_EVERY_N_EPOCH}")
    print("...")
    print("="*80)

    # 再次确保环境变量设置
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

    # 加载配置 - 使用正确的项目代码目录
    config_path = PROJECT_ROOT / "mattergen" / "conf"

    # 使用Hydra风格的配置加载，模拟命令: mattergen-train data_module=mp_20 ~trainer.logger
    @hydra.main(config_path=str(config_path), config_name="default", version_base="1.1")
    def hydra_main(cfg: omegaconf.DictConfig):
        # 应用~trainer.logger (移除logger)
        if hasattr(cfg, 'trainer') and hasattr(cfg.trainer, 'logger'):
            delattr(cfg.trainer, 'logger')

        # 最简化处理：直接清空所有回调，避免任何回调相关问题
        if 'trainer' in cfg and 'callbacks' in cfg.trainer:
            print("移除所有回调以避免回调相关问题")
            cfg.trainer.callbacks = []

        # 应用自定义参数 (等价于 +param=value 命令行参数)
        # 数据模块参数
        cfg.data_module.root_dir = DATASET_PATH
        cfg.data_module.batch_size.train = BATCH_SIZE_TRAIN
        cfg.data_module.batch_size.val = BATCH_SIZE_VAL
        cfg.data_module.batch_size.test = BATCH_SIZE_TEST

        # 训练器参数
        cfg.trainer.accelerator = ACCELERATOR
        cfg.trainer.devices = DEVICES
        cfg.trainer.precision = PRECISION
        cfg.trainer.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES

        # 优化器参数
        cfg.lightning_module.optimizer_partial.lr = LEARNING_RATE

        # 创建完整的配置对象
        schema = OmegaConf.structured(Config)
        final_config = OmegaConf.merge(schema, cfg)

        # 启用自动恢复功能
        OmegaConf.set_struct(final_config, False)  # 临时关闭结构保护
        final_config.auto_resume = True
        OmegaConf.set_struct(final_config, True)  # 重新开启结构保护

        # 打印配置，但跳过可能引起eval解析错误的部分
        try:
            print("当前配置:")
            print(OmegaConf.to_yaml(final_config, resolve=True))
        except Exception as e:
            print(f"配置解析警告: {e}")
            print("跳过详细配置输出，继续训练...")

        print(f"\n使用数据集: {DATA_MODULE_NAME}")
        print(f"数据集路径: {DATASET_PATH}")
        print(f"训练参数: 批大小={BATCH_SIZE_TRAIN}, 学习率={LEARNING_RATE}")
        print(f"梯度累积步数: {ACCUMULATE_GRAD_BATCHES}")

        # 如需使用高级参数，请取消下面的注释
        # print(f"最大训练轮数: {MAX_EPOCHS}")
        # print(f"梯度裁剪值: {GRADIENT_CLIP_VAL}")
        # print(f"验证频率: 每 {CHECK_VAL_EVERY_N_EPOCH} 个epoch验证一次")

        # 运行训练，明确指定save_config=False以避免配置覆盖冲突
        main(final_config, save_config=False)

    # 重要：确保在调用 hydra_main 之前，所需的模块已完全加载
    hydra_main()


if __name__ == "__main__":
    run_training()