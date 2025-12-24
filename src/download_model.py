#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MatterGen模型批量下载脚本

该脚本用于预先批量下载Hugging Face Hub上的模型到本地缓存，
避免在生成过程中重复下载或因网络问题导致失败。
"""

import os
import sys
from pathlib import Path
import glob

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download, scan_cache_dir
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo


def check_model_in_cache(model_name):
    """
    检查模型是否已在本地缓存中
    
    Args:
        model_name: 预训练模型名称
    
    Returns:
        bool: 模型是否已在缓存中
    """
    try:
        # 扫描Hugging Face缓存目录
        hf_cache_info = scan_cache_dir()
        
        # 检查缓存中是否存在模型的两个关键文件
        expected_files = [
            f"checkpoints/{model_name}/checkpoints/last.ckpt",
            f"checkpoints/{model_name}/config.yaml"
        ]
        
        for repo in hf_cache_info.repos:
            if repo.repo_id == "microsoft/mattergen":
                for revision in repo.revisions:
                    for expected_file in expected_files:
                        if any(file.file_path.endswith(expected_file) for file in revision.files):
                            continue  # 找到该文件
                        else:
                            return False  # 缺少某个文件
                return True  # 所有文件都存在
        
        return False
    except Exception:
        # 如果扫描缓存失败，我们回退到路径检查
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        model_path = Path(hf_home) / "hub" / "models--microsoft--mattergen"
        
        # 检查模型文件是否存在于任何快照目录中
        snapshot_dirs = glob.glob(str(model_path / "snapshots" / "*"))
        
        for snapshot_dir in snapshot_dirs:
            snapshot_path = Path(snapshot_dir) / "checkpoints" / model_name
            last_ckpt_path = snapshot_path / "checkpoints" / "last.ckpt"
            config_path = snapshot_path / "config.yaml"
            
            if last_ckpt_path.exists() and config_path.exists():
                return True  # 找到模型文件
        
        return False


def find_model_paths(model_name):
    """
    查找模型文件的实际路径
    
    Args:
        model_name: 预训练模型名称
    
    Returns:
        tuple: (ckpt_path, config_path) 或 (None, None) 如果未找到
    """
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    model_path = Path(hf_home) / "hub" / "models--microsoft--mattergen"
    
    # 检查模型文件是否存在于任何快照目录中
    snapshot_dirs = glob.glob(str(model_path / "snapshots" / "*"))
    
    for snapshot_dir in snapshot_dirs:
        snapshot_path = Path(snapshot_dir) / "checkpoints" / model_name
        last_ckpt_path = snapshot_path / "checkpoints" / "last.ckpt"
        config_path = snapshot_path / "config.yaml"
        
        if last_ckpt_path.exists() and config_path.exists():
            return str(last_ckpt_path), str(config_path)
    
    return None, None


def download_model(model_name):
    """
    下载指定的预训练模型到本地缓存
    
    Args:
        model_name: 预训练模型名称
    """
    print(f"检查模型: {model_name}")
    
    # 首先检查模型是否已在缓存中
    if check_model_in_cache(model_name):
        print(f"模型 {model_name} 已存在于缓存中，跳过下载")
        ckpt_path, config_path = find_model_paths(model_name)
        if ckpt_path and config_path:
            print(f"  检查点文件: {ckpt_path}")
            print(f"  配置文件: {config_path}")
        return
    
    # 设置环境变量
    os.environ["HF_HOME"] = "/home/wczhou/data_linked/.cache/huggingface"
    
    print(f"开始下载模型: {model_name}")
    
    # 尝试使用镜像端点下载（优先使用，因为官方端点可能无法直连）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"尝试使用镜像端点: {os.environ['HF_ENDPOINT']}")
    
    try:
        ckpt_path = hf_hub_download(
            repo_id="microsoft/mattergen",
            filename=f"checkpoints/{model_name}/checkpoints/last.ckpt"
        )
        
        config_path = hf_hub_download(
            repo_id="microsoft/mattergen", 
            filename=f"checkpoints/{model_name}/config.yaml"
        )
        
        print(f"模型 {model_name} 下载完成")
        print(f"  检查点文件: {ckpt_path}")
        print(f"  配置文件: {config_path}")
        
    except Exception as e:
        print(f"使用镜像端点({os.environ['HF_ENDPOINT']})下载模型 {model_name} 时发生错误: {str(e)}")
        
        # 如果镜像端点失败，可能是由于频率限制，尝试直接连接官方端点
        print(f"尝试使用官方端点: https://huggingface.co")
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
        
        try:
            ckpt_path = hf_hub_download(
                repo_id="microsoft/mattergen",
                filename=f"checkpoints/{model_name}/checkpoints/last.ckpt"
            )
            
            config_path = hf_hub_download(
                repo_id="microsoft/mattergen", 
                filename=f"checkpoints/{model_name}/config.yaml"
            )
            
            print(f"模型 {model_name} 下载完成")
            print(f"  检查点文件: {ckpt_path}")
            print(f"  配置文件: {config_path}")
            
        except Exception as e2:
            # 检查本地是否存在模型文件，因为即使下载失败，文件可能已存在于本地
            ckpt_path, config_path = find_model_paths(model_name)
            
            if ckpt_path and config_path:
                print(f"模型 {model_name} 已存在于本地路径，跳过下载")
                print(f"  检查点文件: {ckpt_path}")
                print(f"  配置文件: {config_path}")
            else:
                print(f"模型 {model_name} 在本地路径中未找到，下载失败")
                expected_ckpt_path = f"/home/wczhou/data_linked/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/*/checkpoints/{model_name}/checkpoints/last.ckpt"
                expected_config_path = f"/home/wczhou/data_linked/.cache/huggingface/hub/models--microsoft--mattergen/snapshots/*/checkpoints/{model_name}/config.yaml"
                print(f"  预期路径: {expected_ckpt_path}")
                print(f"  预期路径: {expected_config_path}")
                raise


def main():
    """主函数，批量下载模型"""
    
    # 在这里手动修改需要下载的模型列表
    models_to_download = [
        "mattergen_base",
        # "chemical_system",  # 取消注释以下载此模型
        # "space_group",      # 取消注释以下载此模型
        # "dft_mag_density",  # 取消注释以下载此模型
        # "dft_band_gap",     # 取消注释以下载此模型
        # "ml_bulk_modulus",  # 取消注释以下载此模型
        # "dft_mag_density_hhi_score",  # 取消注释以下载此模型
        # "chemical_system_energy_above_hull",  # 取消注释以下载此模型
        # "mp_20_base",       # 取消注释以下载此模型
    ]
    
    print(f"开始批量下载模型，共 {len(models_to_download)} 个")
    print(f"缓存目录: {os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))}")
    
    for model_name in models_to_download:
        try:
            download_model(model_name)
            print(f"✓ 模型 {model_name} 处理成功\n")
        except Exception as e:
            print(f"✗ 模型 {model_name} 处理失败: {e}\n")
            # 继续处理其他模型
            continue
    
    print("批量处理任务完成！")


if __name__ == "__main__":
    main()