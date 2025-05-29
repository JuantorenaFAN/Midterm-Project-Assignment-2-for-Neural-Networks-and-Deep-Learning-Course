#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.events import EventStorage
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as transforms


class TensorboardLogger:
    """Tensorboard logger for tracking training and evaluation metrics."""
    
    def __init__(self, log_dir, model_name):
        """
        Initialize the Tensorboard logger.
        
        Args:
            log_dir (str): Directory to save Tensorboard logs
            model_name (str): Name of the model (used as subdirectory)
        """
        self.log_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.model_name = model_name
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value to Tensorboard."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag_value_dict, step):
        """Log multiple scalar values to Tensorboard."""
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag, image, step):
        """
        Log an image to Tensorboard.
        
        Args:
            tag (str): Tag for the image
            image (ndarray or tensor): Image to log (RGB format)
            step (int): Global step value
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
        
        if image.dtype != np.uint8:
            # Convert float image to uint8 if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to PIL Image and then to tensor
        image_pil = Image.fromarray(image)
        transform = transforms.ToTensor()
        image_tensor = transform(image_pil)
        
        self.writer.add_image(tag, image_tensor, step)
    
    def log_figure(self, tag, figure, step):
        """
        Log a matplotlib figure to Tensorboard.
        
        Args:
            tag (str): Tag for the figure
            figure (matplotlib.figure.Figure): Figure to log
            step (int): Global step value
        """
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = transforms.ToTensor()(image)
        self.writer.add_image(tag, image_tensor, step)
        plt.close(figure)
    
    def log_histogram(self, tag, values, step, bins='auto'):
        """Log a histogram to Tensorboard."""
        self.writer.add_histogram(tag, values, step, bins=bins)
    
    def log_pr_curve(self, tag, precision, recall, step):
        """
        Log precision-recall curve to Tensorboard.
        
        Args:
            tag (str): Tag for the PR curve
            precision (array): Precision values
            recall (array): Recall values
            step (int): Global step value
        """
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {tag}')
        ax.grid(True)
        
        self.log_figure(f"{tag}/pr_curve", fig, step)
    
    def log_detection_examples(self, tag, images, predictions, step, metadata=None, max_images=5):
        """
        Log detection examples to Tensorboard.
        
        Args:
            tag (str): Tag for the detection examples
            images (list): List of images
            predictions (list): List of predictions
            step (int): Global step value
            metadata (Metadata): Detectron2 metadata
            max_images (int): Maximum number of images to log
        """
        from detectron2.utils.visualizer import Visualizer, ColorMode
        
        num_images = min(len(images), max_images)
        for i in range(num_images):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            
            # Convert to RGB if needed
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            
            # Scale to 0-255 if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Create visualizer
            visualizer = Visualizer(
                image,
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW,
            )
            
            # Draw predictions
            vis_output = visualizer.draw_instance_predictions(predictions[i])
            vis_image = vis_output.get_image()
            
            # Log to Tensorboard
            self.log_image(f"{tag}/detection_{i}", vis_image, step)
    
    def close(self):
        """Close the Tensorboard writer."""
        self.writer.close()


class DetectronEventWriter:
    """
    Writer that translates Detectron2 EventStorage events to TensorboardLogger.
    This class helps bridge the gap between Detectron2's events and Tensorboard.
    """
    
    def __init__(self, tensorboard_logger):
        """
        Initialize the event writer.
        
        Args:
            tensorboard_logger (TensorboardLogger): Tensorboard logger
        """
        self.tb_logger = tensorboard_logger
    
    def write(self, storage):
        """
        Write events from storage to Tensorboard.
        
        Args:
            storage (EventStorage): Detectron2 EventStorage object
        """
        # Log all scalars
        for k, v in storage.latest().items():
            self.tb_logger.log_scalar(k, v, storage.iter)
        
        # Log histograms
        for k, v in storage.histories().items():
            if k.endswith("/hist"):
                # Extract tag name by removing "/hist" suffix
                tag = k[:-5]
                self.tb_logger.log_histogram(tag, np.array(v.values()), storage.iter)
        
        # Log images
        for k, img_tensor in storage.vis_imgs.items():
            self.tb_logger.log_image(k, img_tensor, storage.iter)


def setup_tensorboard(output_dir, model_name):
    """
    Set up Tensorboard logger for Detectron2.
    
    Args:
        output_dir (str): Output directory
        model_name (str): Model name
        
    Returns:
        TensorboardLogger: Tensorboard logger
    """
    log_dir = os.path.join(output_dir, "tensorboard")
    return TensorboardLogger(log_dir, model_name) 