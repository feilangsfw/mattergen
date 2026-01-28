#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行MatterGen训练任务的便捷脚本 - 支持多模型类型
此脚本支持多种模型类型，通过config_name参数选择不同配置
等价于运行:
- mattergen-train data_module=mp_20 ~trainer.logger (默认)
- mattergen-train --config-name=csp data_module=mp_20 ~trainer.logger (CSP模型)
"""

import os
import sys
from pathlib import Path
import argparse

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

def run_training(CONFIG_NAME=None, DATA_MODULE_NAME=None,
                 # 基础参数
                 BATCH_SIZE_TRAIN=None, BATCH_SIZE_VAL=None, BATCH_SIZE_TEST=None,
                 ACCELERATOR=None, DEVICES=None, PRECISION=None,
                 ACCUMULATE_GRAD_BATCHES=None, LEARNING_RATE=None):
    """
    运行训练任务，支持多种配置

    Args:
        CONFIG_NAME: 配置名称 ("default", "csp", "finetune" 等)
        DATA_MODULE_NAME: 数据模块名称
        # 基础参数
        BATCH_SIZE_TRAIN: 训练批次大小
        BATCH_SIZE_VAL: 验证批次大小
        BATCH_SIZE_TEST: 测试批次大小
        ACCELERATOR: 训练设备类型
        DEVICES: 设备数量
        PRECISION: 训练精度
        ACCUMULATE_GRAD_BATCHES: 梯度累积步数
        LEARNING_RATE: 学习率
    """
    # 从命令行参数获取配置名称和其他参数，如果没有提供则使用命令行参数
    args = parse_args()

    # 配置参数 - 这些是你可以根据需要修改的参数
    # 这些参数等价于在命令行中使用 +param=value 的形式覆盖YAML配置
    CONFIG_NAME = CONFIG_NAME or args.config_name              # 配置名称 ("default", "csp", "finetune" 等)
    DATA_MODULE_NAME = DATA_MODULE_NAME or args.data_module         # 等价于命令行: data_module=mp_20
    DATASET_PATH = str(PROJECT_ROOT / "datasets" / "cache" / DATA_MODULE_NAME)  # 数据集路径

    # 数据模块参数 (来自 conf/data_module/mp_20.yaml)
    BATCH_SIZE_TRAIN = BATCH_SIZE_TRAIN or args.batch_size_train    # 训练批处理大小，降低以节省内存
    BATCH_SIZE_VAL = BATCH_SIZE_VAL or args.batch_size_val        # 验证批处理大小，降低以节省内存
    BATCH_SIZE_TEST = BATCH_SIZE_TEST or args.batch_size_test      # 测试批处理大小，降低以节省内存

    # 训练器参数 (来自 conf/trainer/default.yaml)
    ACCELERATOR = ACCELERATOR or args.accelerator              # 使用GPU
    DEVICES = DEVICES or args.devices                      # 自动检测可用GPU
    PRECISION = PRECISION or args.precision                  # 精度 (32表示FP32)
    ACCUMULATE_GRAD_BATCHES = ACCUMULATE_GRAD_BATCHES or args.accumulate_grad_batches  # 梯度累积步数，补偿较小的批处理大小

    # 优化器参数 (来自 conf/lightning_module/default.yaml)
    LEARNING_RATE = LEARNING_RATE or args.learning_rate          # 学习率

    # 可选的高级参数（可根据需要取消注释并调整）
    # MAX_EPOCHS = args.max_epochs              # 最大训练轮数
    # GRADIENT_CLIP_VAL = args.gradient_clip_val # 梯度裁剪值
    # CHECK_VAL_EVERY_N_EPOCH = args.check_val_every_n_epoch  # 每N个epoch验证一次
    # SCHEDULER_FACTOR = args.scheduler_factor  # 学习率衰减因子
    # SCHEDULER_PATIENCE = args.scheduler_patience # 调度器耐心值
    # SCHEDULER_MIN_LR = args.scheduler_min_lr   # 最小学习率
    # HIDDEN_DIM = args.hidden_dim              # 隐藏层维度，降低以节省内存
    # GEMNET_NUM_BLOCKS = args.gemnet_num_blocks   # GemNet块的数量，降低以节省内存
    # GEMNET_CUTOFF = args.gemnet_cutoff        # GemNet截断距离

    print("="*80)
    print("路径变量信息:")
    print(f"PROJECT_ROOT (Path object): {PROJECT_ROOT}")
    print(f"Environment PROJECT_ROOT: {os.environ.get('PROJECT_ROOT', 'NOT SET')}")
    print(f"Environment MATTERGEN_MODELS_PROJECT_ROOT: {os.environ.get('MATTERGEN_MODELS_PROJECT_ROOT', 'NOT SET')}")
    print("="*80)

    print(f"正在使用配置: {CONFIG_NAME}")
    print("此脚本等价于运行以下命令:")
    if CONFIG_NAME == "default":
        print(f"mattergen-train data_module={DATA_MODULE_NAME} ~trainer.logger")
    else:
        print(f"mattergen-train --config-name={CONFIG_NAME} data_module={DATA_MODULE_NAME} ~trainer.logger")

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

    # 使用Hydra风格的配置加载，使用传入的config_name
    @hydra.main(config_path=str(config_path), config_name=CONFIG_NAME, version_base="1.1")
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

        # 禁用auto_resume功能以防止训练重复 - 这是问题的关键
        OmegaConf.set_struct(final_config, False)  # 临时关闭结构保护
        final_config.auto_resume = False  # 将auto_resume设置为False
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


def parse_args():
    parser = argparse.ArgumentParser(description="运行MatterGen训练任务，支持多种配置")
    parser.add_argument('--config-name', type=str, default='default',
                        choices=['default', 'csp', 'finetune'],
                        help='配置名称 (default: default)')
    parser.add_argument('--data-module', type=str, default='mp_20',
                        help='数据模块名称 (default: mp_20)')

    # 基础参数
    parser.add_argument('--batch-size-train', type=int, default=32,
                        help='训练批次大小 (default: 32)')
    parser.add_argument('--batch-size-val', type=int, default=16,
                        help='验证批次大小 (default: 16)')
    parser.add_argument('--batch-size-test', type=int, default=16,
                        help='测试批次大小 (default: 16)')
    parser.add_argument('--accelerator', type=str, default='gpu',
                        help='训练设备类型 (default: gpu)')
    parser.add_argument('--devices', type=str, default='auto',
                        help='设备数量 (default: auto)')
    parser.add_argument('--precision', type=int, default=32,
                        help='训练精度 (default: 32)')
    parser.add_argument('--accumulate-grad-batches', type=int, default=4,
                        help='梯度累积步数 (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='学习率 (default: 0.0001)')

    # 高级参数（注释掉，需要时可启用）
    # parser.add_argument('--max-epochs', type=int, default=900,
    #                     help='最大训练轮数 (default: 900)')
    # parser.add_argument('--gradient-clip-val', type=float, default=0.5,
    #                     help='梯度裁剪值 (default: 0.5)')
    # parser.add_argument('--check-val-every-n-epoch', type=int, default=5,
    #                     help='每N个epoch验证一次 (default: 5)')
    # parser.add_argument('--scheduler-factor', type=float, default=0.6,
    #                     help='学习率衰减因子 (default: 0.6)')
    # parser.add_argument('--scheduler-patience', type=int, default=100,
    #                     help='调度器耐心值 (default: 100)')
    # parser.add_argument('--scheduler-min-lr', type=float, default=1e-6,
    #                     help='最小学习率 (default: 1e-6)')
    # parser.add_argument('--hidden-dim', type=int, default=256,
    #                     help='隐藏层维度 (default: 256)')
    # parser.add_argument('--gemnet-num-blocks', type=int, default=2,
    #                     help='GemNet块的数量 (default: 2)')
    # parser.add_argument('--gemnet-cutoff', type=float, default=7.0,
    #                     help='GemNet截断距离 (default: 7.0)')

    # 解析已知参数，忽略未知参数（如Hydra添加的参数）
    args, unknown = parser.parse_known_args()

    return args


if __name__ == "__main__":
    # 可以直接在此处调用run_training并传递调试参数
    run_training(
        CONFIG_NAME='csp',  # 直接指定使用csp配置
        # BATCH_SIZE_TRAIN=16,  # 直接指定较小的批次大小用于调试
        # LEARNING_RATE=0.001,  # 直接指定较大的学习率用于调试
    )