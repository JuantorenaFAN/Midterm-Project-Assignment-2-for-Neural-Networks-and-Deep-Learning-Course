#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tarfile
import requests
from tqdm import tqdm
import shutil
import subprocess

# URLs for VOC dataset
VOC2007_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
VOC2007_TEST_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOC_DIR = os.path.join(DATA_DIR, "VOCdevkit")


def download_file(url, save_path):
    """Download file from URL with progress bar."""
    print(f"Downloading {url} to {save_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)


def extract_tar(tar_path, extract_path):
    """Extract tar file."""
    print(f"Extracting {tar_path} to {extract_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)


def create_symlink_for_detectron2():
    """Create symbolic link for Detectron2 default dataset path."""
    voc2007_path = os.path.join(VOC_DIR, "VOC2007")
    datasets_dir = os.path.join(BASE_DIR, "datasets")
    datasets_voc_path = os.path.join(datasets_dir, "VOC2007")
    
    # Create datasets directory if it doesn't exist
    if not os.path.exists(datasets_dir):
        print(f"Creating datasets directory at {datasets_dir}")
        os.makedirs(datasets_dir, exist_ok=True)
    
    # Create symbolic link if it doesn't exist
    if not os.path.exists(datasets_voc_path) and os.path.exists(voc2007_path):
        try:
            # Use relative path for the link to make it more portable
            rel_path = os.path.relpath(voc2007_path, datasets_dir)
            print(f"Creating symbolic link from {datasets_voc_path} to {rel_path}")
            os.symlink(rel_path, datasets_voc_path)
            print(f"Created symbolic link from {datasets_voc_path} to {voc2007_path}")
        except Exception as e:
            print(f"Failed to create symbolic link: {e}")
            # Fallback to copy if symlink fails
            try:
                print(f"Falling back to copying files from {voc2007_path} to {datasets_voc_path}")
                shutil.copytree(voc2007_path, datasets_voc_path)
                print(f"Copied VOC2007 data to {datasets_voc_path}")
            except Exception as e2:
                print(f"Failed to copy data: {e2}")
                print("WARNING: Detectron2 may not be able to find the VOC dataset!")


def main():
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download and extract VOC2007 trainval
    voc2007_tar = os.path.join(DATA_DIR, "VOCtrainval_06-Nov-2007.tar")
    if not os.path.exists(voc2007_tar):
        download_file(VOC2007_URL, voc2007_tar)
    extract_tar(voc2007_tar, DATA_DIR)
    
    # Download and extract VOC2007 test
    voc2007_test_tar = os.path.join(DATA_DIR, "VOCtest_06-Nov-2007.tar")
    if not os.path.exists(voc2007_test_tar):
        download_file(VOC2007_TEST_URL, voc2007_test_tar)
    extract_tar(voc2007_test_tar, DATA_DIR)
    
    # Verify the dataset structure
    voc2007_path = os.path.join(VOC_DIR, "VOC2007")
    if os.path.exists(voc2007_path):
        print("VOC2007 dataset downloaded and extracted successfully to:")
        print(voc2007_path)
        
        # Check for essential directories
        for subdir in ["Annotations", "ImageSets", "JPEGImages"]:
            path = os.path.join(voc2007_path, subdir)
            if os.path.exists(path):
                num_files = len(os.listdir(path))
                print(f"  - {subdir}: {num_files} files")
            else:
                print(f"  - {subdir}: MISSING!")
        
        # Create symlink for Detectron2
        create_symlink_for_detectron2()
        
        # 检查ImageSets/Main目录是否存在
        main_dir = os.path.join(voc2007_path, "ImageSets", "Main")
        if not os.path.exists(main_dir):
            print(f"错误: 找不到ImageSets/Main目录: {main_dir}")
            return
            
        # 检查分割文件是否存在
        for split in ["train.txt", "val.txt", "test.txt"]:
            split_file = os.path.join(main_dir, split)
            if not os.path.exists(split_file):
                print(f"错误: 找不到分割文件 {split_file}")
            else:
                with open(split_file, 'r') as f:
                    lines = f.readlines()
                print(f"  - {split}: {len(lines)} 条目")
                
        # 确保JPEGImages目录下确实有图像
        jpeg_dir = os.path.join(voc2007_path, "JPEGImages")
        if os.path.exists(jpeg_dir):
            images = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
            print(f"  - JPEGImages: {len(images)} 图像文件")
            
            # 随机检查几个图像是否可以读取
            if images:
                import random
                sample_images = random.sample(images, min(5, len(images)))
                print("  - 检查样本图像:")
                for img in sample_images:
                    img_path = os.path.join(jpeg_dir, img)
                    file_size = os.path.getsize(img_path)
                    print(f"    {img}: {file_size} bytes")
        
    else:
        print(f"错误: 找不到VOC2007数据集: {voc2007_path}")
        print("请确保VOC数据集已正确下载和解压。")


if __name__ == "__main__":
    main() 