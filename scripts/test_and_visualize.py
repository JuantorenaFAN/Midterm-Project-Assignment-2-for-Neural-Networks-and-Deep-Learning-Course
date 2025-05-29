#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo

from data.voc_dataset import VOC_CLASSES
from utils.utils import set_seed, visualize_detection_results, visualize_proposal_boxes
from utils.utils import visualize_side_by_side, download_test_images


def setup_mask_rcnn_config(model_path):
    """
    Set up configuration for Mask R-CNN.
    
    Args:
        model_path (str): Path to model weights
        
    Returns:
        cfg: Detectron2 config
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 禁用掩码功能，因为VOC数据集没有掩码标注
    cfg.MODEL.MASK_ON = False
    
    # Set data directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voc_dir = os.path.join(root_dir, "data", "VOCdevkit")
    voc2007_dir = os.path.join(voc_dir, "VOC2007")
    
    # Print actual dataset paths for debugging
    print(f"VOC directory: {voc_dir}")
    print(f"VOC2007 directory: {voc2007_dir}")
    print(f"Main directory exists: {os.path.exists(os.path.join(voc2007_dir, 'ImageSets', 'Main'))}")
    
    # Manually set dataset paths for VOC
    cfg.DATASETS.TEST = ("voc_2007_test",)
    
    # Register VOC datasets
    from detectron2.data.datasets import register_pascal_voc
    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    # Clear existing registrations if any
    for split in ["train", "val", "test"]:
        name = f"voc_2007_{split}"
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            print(f"Removed existing registration for {name}")
    
    # Define VOC classes
    voc_classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    
    # Register datasets with explicit paths
    try:
        from detectron2.data.datasets.pascal_voc import load_voc_instances
        
        # 清除之前的注册（如果存在）
        if "voc_2007_test" in DatasetCatalog:
            DatasetCatalog.remove("voc_2007_test")
            print(f"Removed existing registration for voc_2007_test")
        
        # 定义一个帮助函数，避免lambda闭包问题
        def get_voc_test_dicts(dirname=voc2007_dir, classes=voc_classes):
            return load_voc_instances(dirname, "test", class_names=classes)
        
        # 直接注册测试数据集
        DatasetCatalog.register("voc_2007_test", lambda: get_voc_test_dicts())
        MetadataCatalog.get("voc_2007_test").set(
            thing_classes=voc_classes,
            dirname=voc2007_dir,
            year=2007,
            split="test",
        )
        print("VOC test dataset registered successfully")
    except Exception as e:
        print(f"Error registering VOC test dataset: {e}")
    
    # Model parameters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # VOC has 20 classes
    
    # Testing parameters
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg


def setup_sparse_rcnn_config(model_path):
    """
    Set up configuration for Sparse R-CNN.
    
    Args:
        model_path (str): Path to model weights
        
    Returns:
        cfg: Detectron2 config
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/sparse_rcnn_R_50_FPN_100_proposals_3x.yaml"))
    
    # Set data directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voc_dir = os.path.join(root_dir, "data", "VOCdevkit")
    voc2007_dir = os.path.join(voc_dir, "VOC2007")
    
    # We don't need to register the dataset again since it's already registered by the Mask R-CNN config
    
    # Dataset
    cfg.DATASETS.TEST = ("voc_2007_test",)
    
    # Model parameters
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # VOC has 20 classes
    
    # Sparse R-CNN specific parameters
    cfg.MODEL.SPARSE_RCNN.NUM_PROPOSALS = 100
    cfg.MODEL.SPARSE_RCNN.NUM_CLASSES = 20
    
    # Testing parameters
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg


def get_model(cfg):
    """
    Build model from configuration.
    
    Args:
        cfg: Detectron2 config
        
    Returns:
        model: Detectron2 model
    """
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    return model


def evaluate_model(cfg, model, dataset_name):
    """
    Evaluate model on dataset.
    
    Args:
        cfg: Detectron2 config
        model: Detectron2 model
        dataset_name (str): Name of dataset
        
    Returns:
        results: Evaluation results
    """
    evaluator = PascalVOCDetectionEvaluator(dataset_name)
    data_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, data_loader, evaluator)
    return results


def get_test_images(dataset_name, num_images=4):
    """
    Get random test images from dataset.
    
    Args:
        dataset_name (str): Name of dataset
        num_images (int): Number of images to get
        
    Returns:
        list: List of dictionaries with image data
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)
    random.shuffle(dataset_dicts)
    return dataset_dicts[:num_images]


def visualize_mask_rcnn_stages(model, image):
    """
    Visualize Mask R-CNN proposal and final results.
    
    Args:
        model: Mask R-CNN model
        image: Input image
        
    Returns:
        tuple: Proposal visualization, final result visualization
    """
    # Get RPN proposals
    with torch.no_grad():
        features = model.backbone(image)
        proposals, _ = model.proposal_generator(image, features)
    
    # Visualize proposals
    proposal_vis = visualize_proposal_boxes(
        image[0].permute(1, 2, 0).cpu().numpy(),
        proposals[0]
    )
    
    # Get final predictions
    with torch.no_grad():
        predictions = model(image)
    
    # Visualize final results
    metadata = MetadataCatalog.get("voc_2007_test")
    result_vis = visualize_detection_results(
        image[0].permute(1, 2, 0).cpu().numpy(),
        predictions[0]["instances"].to("cpu"),
        metadata
    )
    
    return proposal_vis, result_vis


def visualize_model_comparison(mask_rcnn_model, sparse_rcnn_model, image, metadata):
    """
    Visualize and compare Mask R-CNN and Sparse R-CNN results.
    
    Args:
        mask_rcnn_model: Mask R-CNN model
        sparse_rcnn_model: Sparse R-CNN model
        image: Input image
        metadata: Dataset metadata
        
    Returns:
        tuple: Mask R-CNN visualization, Sparse R-CNN visualization
    """
    # Get Mask R-CNN predictions
    with torch.no_grad():
        mask_rcnn_predictions = mask_rcnn_model(image)
    
    # Visualize Mask R-CNN results
    mask_rcnn_vis = visualize_detection_results(
        image[0].permute(1, 2, 0).cpu().numpy(),
        mask_rcnn_predictions[0]["instances"].to("cpu"),
        metadata
    )
    
    # Get Sparse R-CNN predictions
    with torch.no_grad():
        sparse_rcnn_predictions = sparse_rcnn_model(image)
    
    # Visualize Sparse R-CNN results
    sparse_rcnn_vis = visualize_detection_results(
        image[0].permute(1, 2, 0).cpu().numpy(),
        sparse_rcnn_predictions[0]["instances"].to("cpu"),
        metadata
    )
    
    return mask_rcnn_vis, sparse_rcnn_vis


def visualize_test_dataset_images(mask_rcnn_model, sparse_rcnn_model, output_dir):
    """
    Visualize and compare models on test dataset images.
    
    Args:
        mask_rcnn_model: Mask R-CNN model
        sparse_rcnn_model: Sparse R-CNN model
        output_dir (str): Output directory for visualizations
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations", "test_dataset")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get test images
    test_images = get_test_images("voc_2007_test", num_images=4)
    metadata = MetadataCatalog.get("voc_2007_test")
    
    for i, image_dict in enumerate(test_images):
        # Read image
        image_path = image_dict["file_name"]
        image = read_image(image_path, format="BGR")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1)).to(mask_rcnn_model.device)
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        
        # Visualize Mask R-CNN stages
        proposal_vis, result_vis = visualize_mask_rcnn_stages(mask_rcnn_model, image_tensor[None])
        
        # Save Mask R-CNN stages
        stages_comparison = visualize_side_by_side(
            proposal_vis, result_vis,
            "Mask R-CNN Proposals", "Mask R-CNN Final Results"
        )
        cv2.imwrite(
            os.path.join(vis_dir, f"test_image_{i+1}_mask_rcnn_stages.jpg"),
            cv2.cvtColor(stages_comparison, cv2.COLOR_RGB2BGR)
        )
        
        # Visualize model comparison
        mask_rcnn_vis, sparse_rcnn_vis = visualize_model_comparison(
            mask_rcnn_model, sparse_rcnn_model, image_tensor[None], metadata
        )
        
        # Save model comparison
        model_comparison = visualize_side_by_side(
            mask_rcnn_vis, sparse_rcnn_vis,
            "Mask R-CNN", "Sparse R-CNN"
        )
        cv2.imwrite(
            os.path.join(vis_dir, f"test_image_{i+1}_model_comparison.jpg"),
            cv2.cvtColor(model_comparison, cv2.COLOR_RGB2BGR)
        )
        
        print(f"Saved visualizations for test image {i+1}")


def visualize_external_images(mask_rcnn_model, sparse_rcnn_model, output_dir):
    """
    Visualize and compare models on external images.
    
    Args:
        mask_rcnn_model: Mask R-CNN model
        sparse_rcnn_model: Sparse R-CNN model
        output_dir (str): Output directory for visualizations
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations", "external_images")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Download test images
    test_image_dir = os.path.join(output_dir, "test_images")
    test_image_paths = download_test_images(test_image_dir, num_images=3)
    
    metadata = MetadataCatalog.get("voc_2007_test")
    
    for i, image_path in enumerate(test_image_paths):
        # Read image
        image = read_image(image_path, format="BGR")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.transpose(2, 0, 1)).to(mask_rcnn_model.device)
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        
        # Visualize model comparison
        mask_rcnn_vis, sparse_rcnn_vis = visualize_model_comparison(
            mask_rcnn_model, sparse_rcnn_model, image_tensor[None], metadata
        )
        
        # Save original image
        cv2.imwrite(
            os.path.join(vis_dir, f"external_image_{i+1}_original.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        
        # Save Mask R-CNN results
        cv2.imwrite(
            os.path.join(vis_dir, f"external_image_{i+1}_mask_rcnn.jpg"),
            cv2.cvtColor(mask_rcnn_vis, cv2.COLOR_RGB2BGR)
        )
        
        # Save Sparse R-CNN results
        cv2.imwrite(
            os.path.join(vis_dir, f"external_image_{i+1}_sparse_rcnn.jpg"),
            cv2.cvtColor(sparse_rcnn_vis, cv2.COLOR_RGB2BGR)
        )
        
        # Save model comparison
        model_comparison = visualize_side_by_side(
            mask_rcnn_vis, sparse_rcnn_vis,
            "Mask R-CNN", "Sparse R-CNN"
        )
        cv2.imwrite(
            os.path.join(vis_dir, f"external_image_{i+1}_model_comparison.jpg"),
            cv2.cvtColor(model_comparison, cv2.COLOR_RGB2BGR)
        )
        
        print(f"Saved visualizations for external image {i+1}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test and visualize models on VOC dataset")
    parser.add_argument("--mask-rcnn-model", type=str, required=True, help="Path to Faster R-CNN model weights (kept name for compatibility)")
    parser.add_argument("--sparse-rcnn-model", type=str, required=True, help="Path to Sparse R-CNN model weights")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up Mask R-CNN
    mask_rcnn_cfg = setup_mask_rcnn_config(args.mask_rcnn_model)
    mask_rcnn_model = get_model(mask_rcnn_cfg)
    
    # Set up Sparse R-CNN
    sparse_rcnn_cfg = setup_sparse_rcnn_config(args.sparse_rcnn_model)
    sparse_rcnn_model = get_model(sparse_rcnn_cfg)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate models
    print("Evaluating Mask R-CNN...")
    mask_rcnn_results = evaluate_model(mask_rcnn_cfg, mask_rcnn_model, "voc_2007_test")
    
    print("Evaluating Sparse R-CNN...")
    sparse_rcnn_results = evaluate_model(sparse_rcnn_cfg, sparse_rcnn_model, "voc_2007_test")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "evaluation_results.txt"), "w") as f:
        f.write("Mask R-CNN Results:\n")
        for task, metrics in mask_rcnn_results.items():
            f.write(f"{task}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
        
        f.write("\nSparse R-CNN Results:\n")
        for task, metrics in sparse_rcnn_results.items():
            f.write(f"{task}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
    
    # Visualize test dataset images
    print("Visualizing test dataset images...")
    visualize_test_dataset_images(mask_rcnn_model, sparse_rcnn_model, args.output_dir)
    
    # Visualize external images
    print("Visualizing external images...")
    visualize_external_images(mask_rcnn_model, sparse_rcnn_model, args.output_dir)
    
    print(f"Testing and visualization completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 