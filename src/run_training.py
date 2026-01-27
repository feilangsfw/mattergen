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

# 在任何其他import之前设置环境变量
os.environ["PROJECT_ROOT"] = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# 获取当前脚本的目录，并设置项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 添加项目根目录到sys.path，以便导入mattergen
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.diffusion.config import Config
from mattergen.diffusion.run import main

def run_training():
    # 设置Tensor Core加速
    torch.set_float32_matmul_precision("high")
    
    # 配置参数 - 这些是你可以根据需要修改的参数
    # 这些参数等价于在命令行中使用 +param=value 的形式覆盖YAML配置
    DATA_MODULE_NAME = "mp_20"  # 等价于命令行: data_module=mp_20
    DATASET_PATH = str(PROJECT_ROOT / "datasets" / "cache" / DATA_MODULE_NAME)  # 数据集路径
    
    # 输出目录设置 (来自 default.yaml 中的 hydra.run.dir)
    OUTPUT_DIR = str(PROJECT_ROOT / "outputs")  # 输出目录
    
    # 数据模块参数 (来自 conf/data_module/mp_20.yaml)
    BATCH_SIZE_TRAIN = 32  # 训练批处理大小，降低以节省内存
    BATCH_SIZE_VAL = 16     # 验证批处理大小，降低以节省内存
    BATCH_SIZE_TEST = 16    # 测试批处理大小，降低以节省内存
    
    # 训练器参数 (来自 conf/trainer/default.yaml)
    MAX_EPOCHS = 900        # 最大训练轮数
    ACCELERATOR = "gpu"     # 使用GPU
    DEVICES = "auto"        # 自动检测可用GPU
    NUM_NODES = 1           # 节点数量
    PRECISION = 32          # 精度 (32表示FP32)
    GRADIENT_CLIP_VAL = 0.5 # 梯度裁剪值
    ACCUMULATE_GRAD_BATCHES = 4  # 梯度累积步数，补偿较小的批处理大小
    
    # 优化器参数 (来自 conf/lightning_module/default.yaml)
    LEARNING_RATE = 0.0001  # 学习率
    
    # 调度器参数 (来自 conf/lightning_module/default.yaml)
    SCHEDULER_FACTOR = 0.6  # 学习率衰减因子
    SCHEDULER_PATIENCE = 100 # 调度器耐心值
    SCHEDULER_MIN_LR = 1e-6   # 最小学习率
    
    # 模型架构参数 (来自 conf/lightning_module/diffusion_module/model/mattergen.yaml)
    HIDDEN_DIM = 256        # 隐藏层维度，降低以节省内存
    GEMNET_NUM_BLOCKS = 2   # GemNet块的数量，降低以节省内存
    GEMNET_CUTOFF = 7.0     # GemNet截断距离
    
    # 模型类型配置 (可以切换不同的模型配置)
    DIFFUSION_MODULE_CONFIG = "default"  # 可以是 "default", "csp" 等
    MODEL_CONFIG = "mattergen"  # 模型类型
    CORRUPTION_CONFIG = "default"  # 数据损坏策略
    
    print("="*80)
    print("此脚本等价于运行以下命令:")
    print(f"mattergen-train data_module={DATA_MODULE_NAME} ~trainer.logger")
    print("+data_module.root_dir='{}'".format(DATASET_PATH))
    print(f"+data_module.batch_size.train={BATCH_SIZE_TRAIN}")
    print(f"+data_module.batch_size.val={BATCH_SIZE_VAL}")
    print(f"+data_module.batch_size.test={BATCH_SIZE_TEST}")
    print(f"+trainer.accelerator={ACCELERATOR}")
    print(f"+trainer.devices={DEVICES}")
    print(f"+trainer.max_epochs={MAX_EPOCHS}")
    print(f"+trainer.accumulate_grad_batches={ACCUMULATE_GRAD_BATCHES}")
    print(f"+lightning_module.optimizer_partial.lr={LEARNING_RATE}")
    print("...")
    print("="*80)
    
    # 加载配置
    config_path = MODELS_PROJECT_ROOT / "conf"
    
    # 使用Hydra风格的配置加载，模拟命令: mattergen-train data_module=mp_20 ~trainer.logger
    @hydra.main(config_path=str(config_path), config_name="default", version_base="1.1")
    def hydra_main(cfg: omegaconf.DictConfig):
        # 手动应用命令行样式的覆盖 (等价于命令行参数覆盖)
        # 应用data_module=mp_20
        data_module_cfg = OmegaConf.load(config_path / f"data_module/{DATA_MODULE_NAME}.yaml")
        cfg.data_module = data_module_cfg
        
        # 应用其他模块配置
        # 加载diffusion_module配置
        if DIFFUSION_MODULE_CONFIG != "default":
            dm_cfg = OmegaConf.load(config_path / f"lightning_module/diffusion_module/{DIFFUSION_MODULE_CONFIG}.yaml")
            cfg.lightning_module.diffusion_module = dm_cfg
        
        # 加载模型配置
        if MODEL_CONFIG != "mattergen":
            model_cfg = OmegaConf.load(config_path / f"lightning_module/diffusion_module/model/{MODEL_CONFIG}.yaml")
            cfg.lightning_module.diffusion_module.model = model_cfg
            
        # 加载corruption配置
        if CORRUPTION_CONFIG != "default":
            cor_cfg = OmegaConf.load(config_path / f"lightning_module/diffusion_module/corruption/{CORRUPTION_CONFIG}.yaml")
            cfg.lightning_module.diffusion_module.corruption = cor_cfg
        
        # 应用~trainer.logger (移除logger)
        if hasattr(cfg, 'trainer') and hasattr(cfg.trainer, 'logger'):
            delattr(cfg.trainer, 'logger')
        
        # 应用自定义参数 (等价于 +param=value 命令行参数)
        # 数据模块参数
        cfg.data_module.root_dir = DATASET_PATH
        cfg.data_module.batch_size.train = BATCH_SIZE_TRAIN
        cfg.data_module.batch_size.val = BATCH_SIZE_VAL
        cfg.data_module.batch_size.test = BATCH_SIZE_TEST
        
        # 训练器参数
        cfg.trainer.accelerator = ACCELERATOR
        cfg.trainer.devices = DEVICES
        cfg.trainer.num_nodes = NUM_NODES
        cfg.trainer.precision = PRECISION
        cfg.trainer.max_epochs = MAX_EPOCHS  # 在trainer中设置最大训练轮数
        cfg.trainer.accumulate_grad_batches = ACCUMULATE_GRAD_BATCHES
        cfg.trainer.gradient_clip_val = GRADIENT_CLIP_VAL
        cfg.trainer.check_val_every_n_epoch = 5  # 每5个epoch验证一次
        
        # 优化器参数
        cfg.lightning_module.optimizer_partial.lr = LEARNING_RATE
        
        # 调度器参数
        if hasattr(cfg.lightning_module, 'scheduler_partials'):
            if len(cfg.lightning_module.scheduler_partials) > 0:
                # 确保不包含verbose参数 (新版本PyTorch已移除此参数)
                scheduler_cfg = cfg.lightning_module.scheduler_partials[0]['scheduler']
                if 'verbose' in scheduler_cfg:
                    del scheduler_cfg['verbose']
                
                # 应用自定义调度器参数
                scheduler_cfg['factor'] = SCHEDULER_FACTOR
                scheduler_cfg['patience'] = SCHEDULER_PATIENCE
                scheduler_cfg['min_lr'] = SCHEDULER_MIN_LR

        # 模型架构参数
        if hasattr(cfg.lightning_module, 'diffusion_module'):
            if hasattr(cfg.lightning_module.diffusion_module, 'model'):
                # 应用GemNet-T模型参数
                if hasattr(cfg.lightning_module.diffusion_module.model, 'hidden_dim'):
                    cfg.lightning_module.diffusion_module.model.hidden_dim = HIDDEN_DIM
                    
                if hasattr(cfg.lightning_module.diffusion_module.model, 'gemnet'):
                    if hasattr(cfg.lightning_module.diffusion_module.model.gemnet, 'num_blocks'):
                        cfg.lightning_module.diffusion_module.model.gemnet.num_blocks = GEMNET_NUM_BLOCKS
                        
                    if hasattr(cfg.lightning_module.diffusion_module.model.gemnet, 'cutoff'):
                        cfg.lightning_module.diffusion_module.model.gemnet.cutoff = GEMNET_CUTOFF

        cfg.auto_resume = True  # 自动恢复训练
        
        # 设置输出目录
        os.environ["OUTPUT_DIR"] = OUTPUT_DIR
        
        # 创建完整的配置对象
        schema = OmegaConf.structured(Config)
        final_config = OmegaConf.merge(schema, cfg)
        OmegaConf.set_readonly(final_config, True)  # 设置为只读
        
        print("当前配置:")
        print(OmegaConf.to_yaml(final_config, resolve=True))
        
        print(f"\n使用数据集: {DATA_MODULE_NAME}")
        print(f"数据集路径: {DATASET_PATH}")
        print(f"输出目录: {OUTPUT_DIR}")
        print(f"训练参数: 批大小={BATCH_SIZE_TRAIN}, 最大轮数={MAX_EPOCHS}, 学习率={LEARNING_RATE}")
        print(f"模型参数: 隐藏维度={HIDDEN_DIM}, GemNet块数={GEMNET_NUM_BLOCKS}")
        print(f"梯度累积步数: {ACCUMULATE_GRAD_BATCHES}")
        
        # 运行训练
        main(final_config)

    hydra_main()


if __name__ == "__main__":
    run_training()