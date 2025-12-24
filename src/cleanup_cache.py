#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理MatterGen项目缓存的脚本
"""

import os
import shutil
from pathlib import Path


def cleanup_hf_cache():
    """清理Hugging Face Hub缓存"""
    print("开始清理Hugging Face Hub缓存...")
    
    # Hugging Face Hub缓存路径
    hf_cache_path = Path("/home/wczhou/data_linked/.cache/huggingface/hub/models--microsoft--mattergen")
    
    if not hf_cache_path.exists():
        print(f"缓存路径不存在: {hf_cache_path}")
        return
    
    snapshots_path = hf_cache_path / "snapshots"
    if not snapshots_path.exists():
        print(f"快照路径不存在: {snapshots_path}")
        return
    
    snapshots = [d for d in snapshots_path.iterdir() if d.is_dir()]
    print(f"发现 {len(snapshots)} 个快照:")
    for snapshot in snapshots:
        checkpoints_path = snapshot / "checkpoints"
        if checkpoints_path.exists():
            subdirs = [d.name for d in checkpoints_path.iterdir() if d.is_dir()]
            print(f"  - {snapshot.name}: 包含 {subdirs}")
    
    # 保留包含更多模型的快照，删除其他快照
    if len(snapshots) > 1:
        snapshots_with_counts = []
        for snapshot in snapshots:
            checkpoints_path = snapshot / "checkpoints"
            if checkpoints_path.exists():
                subdirs = [d for d in checkpoints_path.iterdir() if d.is_dir()]
                snapshots_with_counts.append((snapshot, len(subdirs)))
        
        # 按子目录数量排序，保留最多的
        snapshots_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"保留快照 {snapshots_with_counts[0][0].name} (包含 {snapshots_with_counts[0][1]} 个模型)")
        
        for snapshot, count in snapshots_with_counts[1:]:
            print(f"删除快照 {snapshot.name} (包含 {count} 个模型)")
            try:
                shutil.rmtree(snapshot)
                print(f"  - 已删除: {snapshot}")
            except Exception as e:
                print(f"  - 删除失败: {e}")
    else:
        print("只有一个快照，无需清理")
    
    print("Hugging Face Hub缓存清理完成")


def cleanup_project_cache():
    """清理项目本地缓存"""
    print("\n开始清理项目本地缓存...")
    
    project_cache_path = Path("/home/wczhou/data_linked/mattergen/.cache")
    
    if not project_cache_path.exists():
        print(f"项目缓存路径不存在: {project_cache_path}")
        return
    
    # 清理下载缓存
    download_cache_path = project_cache_path / "huggingface" / "download"
    if download_cache_path.exists():
        print(f"删除下载缓存: {download_cache_path}")
        try:
            shutil.rmtree(download_cache_path)
            print("  - 已删除下载缓存")
        except Exception as e:
            print(f"  - 删除下载缓存失败: {e}")
    
    print("项目本地缓存清理完成")


def main():
    """主函数"""
    print("开始清理缓存...")
    
    cleanup_hf_cache()
    cleanup_project_cache()
    
    print("\n缓存清理完成！")


if __name__ == "__main__":
    main()

