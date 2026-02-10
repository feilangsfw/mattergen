#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行MatterGen训练任务的便捷脚本（重构版本）
此脚本等价于运行:
mattergen-train data_module=mp_20 ~trainer.logger
但允许通过Python代码直接修改参数，而不需要修改YAML文件
"""

import os
import sys
from pathlib import Path

# 在任何其他import之前设置环境变量，使用Path对象然后转换为字符串
PROJECT_ROOT = Path(__file__).parent.parent  # /home/wczhou/data_linked/projects/mattergen

def initialize_environment():
    """初始化环境变量和路径设置"""
    # 关键：在导入mattergen模块前设置环境变量
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["MATTERGEN_MODELS_PROJECT_ROOT"] = str(PROJECT_ROOT)
    # 新增：提前设置 OUTPUT_DIR 环境变量
    os.environ["OUTPUT_DIR"] = str(PROJECT_ROOT / "outputs")

    # 重要：预设PYTHONPATH，确保Python能找到正确的mattergen模块
    sys.path.insert(0, str(PROJECT_ROOT))

# 在程序启动时立即初始化环境（只执行一次）
initialize_environment()

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf
import torch.distributed as dist

# 现在导入mattergen相关模块
from mattergen.diffusion.config import Config
from mattergen.diffusion.run import main
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT

print(f"[DEBUG] 程序开始执行 - PID: {os.getpid()}")

# 官方的Hydra装饰器配置加载，模拟命令: mattergen-train data_module=mp_20 ~trainer.logger
@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="csp", version_base="1.1"
)
def mattergen_main(cfg: omegaconf.DictConfig):
    print(f"[DEBUG] 进入mattergen_main函数 - PID: {os.getpid()}")

    try:
        # Tensor Core acceleration (leads to ~2x speed-up during訓練)
        torch.set_float32_matmul_precision("high")

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

        # 显式设置 max_epochs 为 5（调试用途）
        cfg.data_module.max_epochs = 3

        # 训练器参数
        cfg.trainer.accelerator = ACCELERATOR
        cfg.trainer.devices = DEVICES
        cfg.trainer.precision = PRECISION
        cfg.trainer.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES

        # 优化器参数
        cfg.lightning_module.optimizer_partial.lr = LEARNING_RATE

        # 简化处理：只合并配置，不修改Hydra内部配置
        schema = OmegaConf.structured(Config)
        final_config = OmegaConf.merge(schema, cfg)
        OmegaConf.set_readonly(final_config, True)  # should not be written to

        # 打印配置，验证设置是否正确
        try:
            print("当前配置:")
            resolved_config = OmegaConf.to_yaml(final_config, resolve=True)
            print(resolved_config)
        except Exception as e:
            print(f"配置解析警告: {e}")
            print("跳过详细配置输出，继续训练...")

        print(f"\n使用数据集: {DATA_MODULE_NAME}")
        print(f"数据集路径: {DATASET_PATH}")
        print(f"训练参数: 批大小={BATCH_SIZE_TRAIN}, 学习率={LEARNING_RATE}")
        print(f"梯度累积步数: {ACCUMULATE_GRAD_BATCHES}")

        # 直接调用main函数，不传递额外的overrides
        main(final_config)
    finally:
        # 清理分布式训练资源
        cleanup_distributed_resources()

def cleanup_distributed_resources():
    """清理分布式训练资源，避免资源泄漏警告"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("成功清理分布式训练资源")
    except Exception as e:
        print(f"清理分布式资源时出现警告: {e}")
        print("这通常是正常现象，不会影响训练结果")

def run_training():
    print(f"[DEBUG] 进入run_training函数 - PID: {os.getpid()}")

    global DATASET_PATH, BATCH_SIZE_TRAIN, BATCH_SIZE_VAL, BATCH_SIZE_TEST
    global ACCELERATOR, DEVICES, PRECISION, ACCUMULATE_GRAD_BATCHES, LEARNING_RATE
    global DATA_MODULE_NAME

    # 配置参数 - 这些是你可以根据需要修改的参数
    DATA_MODULE_NAME = "mp_20"  # 等价于命令行: data_module=mp_20
    DATASET_PATH = str(PROJECT_ROOT / "datasets" / "cache" / DATA_MODULE_NAME)

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
    print("="*80)

    # 直接调用主函数，让Hydra装饰器处理配置加载
    print(f"[DEBUG] 即将调用mattergen_main - PID: {os.getpid()}")
    mattergen_main()


if __name__ == "__main__":
    print(f"[DEBUG] 进入__main__块 - PID: {os.getpid()}")

    print("="*80)
    print("路径变量信息:")
    print(f"PROJECT_ROOT (Path object): {PROJECT_ROOT}")
    print(f"Environment PROJECT_ROOT: {os.environ.get('PROJECT_ROOT', 'NOT SET')}")
    print(f"Environment MATTERGEN_MODELS_PROJECT_ROOT: {os.environ.get('MATTERGEN_MODELS_PROJECT_ROOT', 'NOT SET')}")
    print("="*80)
    try:
        print(f"[DEBUG] 即将调用run_training - PID: {os.getpid()}")
        run_training()
        print(f"[DEBUG] run_training执行完毕 - PID: {os.getpid()}")
    except KeyboardInterrupt:
        print("\n訓練被用户中断")
    except Exception as e:
        print(f"\n訓練过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源被清理
        cleanup_distributed_resources()
        print("程序正常退出")