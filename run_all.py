#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import datetime

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run all steps for training and evaluating models")
    parser.add_argument("--max-iter", type=int, default=5000, help="Maximum iterations for training")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--mask-rcnn-model", type=str, default=None, help="Path to Mask R-CNN model weights")
    parser.add_argument("--sparse-rcnn-model", type=str, default=None, help="Path to Sparse R-CNN model weights")
    
    return parser.parse_args()


def run_command(command, description=None):
    """Run a command and print the output."""
    if description:
        print(f"\n{description}")
    
    print(f"Running command: {command}")
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="", flush=True)
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Command failed with return code {return_code}")
            return False
        
        return True
    except Exception as e:
        print(f"Error executing command: {e}")
        return False


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确保当前目录结构正确
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 检查和创建必要的目录结构
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    voc_dir = os.path.join(data_dir, "VOCdevkit")
    if not os.path.exists(voc_dir):
        os.makedirs(voc_dir, exist_ok=True)
    
    # 创建datasets目录，用于Detectron2默认路径
    datasets_dir = os.path.join(current_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # 检查VOC数据集是否已下载
    voc2007_dir = os.path.join(voc_dir, "VOC2007")
    
    # 如果没有VOC数据集，下载它
    if not os.path.exists(voc2007_dir) or not os.path.isdir(voc2007_dir) or not os.path.exists(os.path.join(voc2007_dir, "ImageSets", "Main", "test.txt")):
        print(f"VOC数据集不完整或不存在，开始下载...")
        success = run_command(
            "python scripts/download_voc.py",
            "下载和解压VOC数据集"
        )
        if not success:
            print("下载VOC数据集失败，退出。")
            return
    else:
        print(f"找到VOC数据集: {voc2007_dir}")
    
    # 在datasets目录创建VOC2007的符号链接
    datasets_voc_path = os.path.join(datasets_dir, "VOC2007")
    if not os.path.exists(datasets_voc_path):
        try:
            # 尝试创建符号链接
            os.symlink(os.path.relpath(voc2007_dir, datasets_dir), datasets_voc_path)
            print(f"创建符号链接 {datasets_voc_path} -> {voc2007_dir}")
        except Exception as e:
            print(f"创建符号链接失败: {e}")
            try:
                # 符号链接失败则尝试复制
                import shutil
                print(f"尝试复制数据目录 {voc2007_dir} -> {datasets_voc_path}")
                shutil.copytree(voc2007_dir, datasets_voc_path)
                print(f"复制VOC2007数据到 {datasets_voc_path}")
            except Exception as e2:
                print(f"复制数据失败: {e2}")
                print("无法创建必要的数据目录，请手动设置。")
                return
    
    # 检查数据集结构是否完整
    for split in ["train.txt", "val.txt", "test.txt"]:
        original_path = os.path.join(voc2007_dir, "ImageSets", "Main", split)
        linked_path = os.path.join(datasets_voc_path, "ImageSets", "Main", split)
        
        if not os.path.exists(original_path):
            print(f"警告: 找不到原始分割文件 {original_path}")
        else:
            print(f"原始分割文件存在: {original_path}")
            
        if not os.path.exists(linked_path) and os.path.exists(original_path):
            print(f"警告: 链接分割文件不存在 {linked_path}")
            
    # 训练模型（如果不是仅评估模式）
    if not args.eval_only:
        # 训练Mask R-CNN
        mask_rcnn_command = (
            f"python scripts/train_mask_rcnn.py "
            f"--batch-size {args.batch_size} "
            f"--max-iter {args.max_iter} "
            f"--output-dir {args.output_dir} "
            f"--eval-period 0"  # 禁用训练期间的评估
        )
        success = run_command(mask_rcnn_command, "训练Mask R-CNN")
        if not success:
            print("训练Mask R-CNN失败，退出。")
            return
        
        # 找到最新的Mask R-CNN模型目录
        mask_rcnn_dirs = [
            os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir)
            if d.startswith("mask_rcnn_") and os.path.isdir(os.path.join(args.output_dir, d))
        ]
        if mask_rcnn_dirs:
            mask_rcnn_dir = sorted(mask_rcnn_dirs)[-1]
            mask_rcnn_model = os.path.join(mask_rcnn_dir, "model_best.pth")
            if not os.path.exists(mask_rcnn_model):
                mask_rcnn_model = os.path.join(mask_rcnn_dir, "model_final.pth")
                if not os.path.exists(mask_rcnn_model):
                    print(f"无法在 {mask_rcnn_dir} 中找到模型权重")
                    return
        else:
            print("找不到Mask R-CNN模型，请检查训练输出。")
            return
        
        # 训练Sparse R-CNN
        sparse_rcnn_command = (
            f"python scripts/train_sparse_rcnn.py "
            f"--batch-size {args.batch_size} "
            f"--max-iter {args.max_iter} "
            f"--output-dir {args.output_dir} "
            f"--eval-period 0"  # 禁用训练期间的评估
        )
        success = run_command(sparse_rcnn_command, "训练Sparse R-CNN")
        if not success:
            print("训练Sparse R-CNN失败，退出。")
            return
        
        # 找到最新的Sparse R-CNN模型目录
        sparse_rcnn_dirs = [
            os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir)
            if d.startswith("sparse_rcnn_") and os.path.isdir(os.path.join(args.output_dir, d))
        ]
        if sparse_rcnn_dirs:
            sparse_rcnn_dir = sorted(sparse_rcnn_dirs)[-1]
            sparse_rcnn_model = os.path.join(sparse_rcnn_dir, "model_best.pth")
            if not os.path.exists(sparse_rcnn_model):
                sparse_rcnn_model = os.path.join(sparse_rcnn_dir, "model_final.pth")
                if not os.path.exists(sparse_rcnn_model):
                    print(f"无法在 {sparse_rcnn_dir} 中找到模型权重")
                    return
        else:
            print("找不到Sparse R-CNN模型，请检查训练输出。")
            return
    else:
        # 使用提供的模型路径进行评估
        mask_rcnn_model = args.mask_rcnn_model
        sparse_rcnn_model = args.sparse_rcnn_model
        
        if not mask_rcnn_model or not sparse_rcnn_model:
            print("在仅评估模式下，必须提供Mask R-CNN和Sparse R-CNN模型的路径。")
            return
        
        if not os.path.exists(mask_rcnn_model) or not os.path.exists(sparse_rcnn_model):
            print(f"一个或两个模型文件不存在: {mask_rcnn_model}, {sparse_rcnn_model}")
            return
    
    # 测试和可视化模型
    test_command = (
        f"python scripts/test_and_visualize.py "
        f"--mask-rcnn-model {mask_rcnn_model} "
        f"--sparse-rcnn-model {sparse_rcnn_model} "
        f"--output-dir {args.output_dir}"
    )
    success = run_command(test_command, "测试和可视化模型")
    if not success:
        print("测试和可视化模型失败。")
        return
    
    print(f"\n所有步骤已成功完成。结果保存在 {args.output_dir}")
    print("\n查看Tensorboard日志，运行:")
    print(f"tensorboard --logdir {os.path.join(args.output_dir, 'tensorboard')}")


if __name__ == "__main__":
    main() 