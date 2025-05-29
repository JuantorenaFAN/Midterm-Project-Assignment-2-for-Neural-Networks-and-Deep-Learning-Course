#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(is_train=True):
    """Get data transforms for training or evaluation."""
    if is_train:
        augmentations = [
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.ResizeShortestEdge(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ]
    else:
        augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
        ]
    return augmentations


def get_evaluator(dataset_name, output_dir):
    """Get appropriate evaluator for dataset."""
    if "voc" in dataset_name.lower():
        return PascalVOCDetectionEvaluator(dataset_name)
    else:
        return COCOEvaluator(dataset_name, output_dir=output_dir)


def visualize_detection_results(image, instances, metadata, scale=1.0):
    """
    Visualize detection results with custom visualization.
    
    Args:
        image (ndarray): Image in RGB format
        instances: Detectron2 Instances object
        metadata: Detectron2 Metadata object
        scale (float): Scale factor for visualization
        
    Returns:
        ndarray: Visualization image
    """
    visualizer = Visualizer(
        image,
        metadata=metadata,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION,
    )
    vis_output = visualizer.draw_instance_predictions(instances)
    return vis_output.get_image()


def visualize_proposal_boxes(image, proposals, scale=1.0):
    """
    Visualize proposal boxes from RPN.
    
    Args:
        image (ndarray): Image in RGB format
        proposals: Detectron2 Instances object with proposal boxes
        scale (float): Scale factor for visualization
        
    Returns:
        ndarray: Visualization image
    """
    fig, ax = plt.subplots(1, figsize=(image.shape[1] / 100, image.shape[0] / 100))
    ax.imshow(image)
    
    # Get proposal boxes (top 20 for visibility)
    if len(proposals.proposal_boxes) > 20:
        scores = proposals.objectness_logits
        indices = torch.argsort(scores, descending=True)[:20]
        boxes = proposals.proposal_boxes[indices].tensor.cpu().numpy()
    else:
        boxes = proposals.proposal_boxes.tensor.cpu().numpy()
    
    # Draw boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle(
            (x1, y1), width, height,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return vis_image


def visualize_side_by_side(image1, image2, title1="", title2=""):
    """
    Visualize two images side by side.
    
    Args:
        image1, image2 (ndarray): Images to visualize
        title1, title2 (str): Titles for images
        
    Returns:
        ndarray: Combined visualization image
    """
    # Ensure same height for both images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    if h1 != h2:
        # Resize to match heights
        scale = h1 / h2
        new_w2 = int(w2 * scale)
        image2 = cv2.resize(image2, (new_w2, h1))
    
    # Create combined image
    combined = np.hstack([image1, image2])
    
    # Add titles
    if title1 or title2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Create space for titles
        title_space = 40
        combined_with_title = np.zeros((combined.shape[0] + title_space, combined.shape[1], 3), dtype=np.uint8)
        combined_with_title[title_space:, :, :] = combined
        
        # Add titles
        if title1:
            cv2.putText(
                combined_with_title, title1, (10, 30),
                font, font_scale, (255, 255, 255), thickness
            )
        
        if title2:
            text_size = cv2.getTextSize(title2, font, font_scale, thickness)[0]
            cv2.putText(
                combined_with_title, title2, (w1 + 10, 30),
                font, font_scale, (255, 255, 255), thickness
            )
        
        return combined_with_title
    
    return combined


def download_test_images(save_dir="test_images", num_images=3):
    """
    Download test images from the internet that contain VOC classes.
    
    Args:
        save_dir (str): Directory to save images
        num_images (int): Number of images to download
        
    Returns:
        list: Paths to downloaded images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # URLs of images containing VOC classes
    image_urls = [
        "https://farm4.staticflickr.com/3693/9472793441_b7822c4929_z.jpg",  # dog and person
        "https://farm8.staticflickr.com/7258/7477750178_c30c85bd7a_z.jpg",  # car
        "https://farm9.staticflickr.com/8251/8631233587_c77fb9b626_z.jpg",  # bicycle
        "https://live.staticflickr.com/8400/8662163689_65d3945cb5_z.jpg",   # cat
        "https://farm1.staticflickr.com/7/5959537_ee62d99a31_z.jpg",        # person and chair
    ]
    
    if num_images > len(image_urls):
        num_images = len(image_urls)
    
    selected_urls = image_urls[:num_images]
    image_paths = []
    
    for i, url in enumerate(selected_urls):
        save_path = os.path.join(save_dir, f"test_image_{i+1}.jpg")
        
        # Skip if file already exists
        if os.path.exists(save_path):
            image_paths.append(save_path)
            continue
        
        try:
            import requests
            response = requests.get(url)
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            image_paths.append(save_path)
            print(f"Downloaded image to {save_path}")
        except Exception as e:
            print(f"Failed to download image from {url}: {e}")
    
    return image_paths 