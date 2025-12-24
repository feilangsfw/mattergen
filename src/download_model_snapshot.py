#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用snapshot_download下载MatterGen模型的脚本
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_model_snapshot(model_name):
    """
    使用snapshot_download下载模型到本地缓存

    Args:
        model_name: 预训练模型名称
    """
    print(f"开始下载模型: {model_name}")
    
    # 设置环境变量
    os.environ["HF_HOME"] = "/home/wczhou/data_linked/.cache/huggingface"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    print(f"使用镜像端点: {os.environ['HF_ENDPOINT']}")
    print(f"缓存目录: {os.environ['HF_HOME']}")
    
    try:
        # 使用snapshot_download下载整个仓库
        model_path = snapshot_download(
            repo_id="microsoft/mattergen",
            allow_patterns=[f"checkpoints/{model_name}/**"],
        )
        
        print(f"模型 {model_name} 下载完成")
        print(f"存储路径: {model_path}")
        
    except Exception as e:
        print(f"下载模型 {model_name} 时发生错误: {str(e)}")
        raise


def main():
    """主函数，下载模型"""
    
    # 指定要下载的模型
    model_to_download = "mp_20_base"
    
    print(f"开始下载模型: {model_to_download}")
    
    try:
        download_model_snapshot(model_to_download)
        print(f"✓ 模型 {model_to_download} 下载成功")
    except Exception as e:
        print(f"✗ 模型 {model_to_download} 下载失败: {e}")


if __name__ == "__main__":
    main()