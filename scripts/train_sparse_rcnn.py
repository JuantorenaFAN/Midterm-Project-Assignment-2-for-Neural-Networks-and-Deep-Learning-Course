#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.utils.events import EventStorage
from detectron2 import model_zoo

from utils.utils import set_seed, get_evaluator
from utils.tensorboard_logger import setup_tensorboard, DetectronEventWriter


class ValidationHook(HookBase):
    """
    Hook for running validation during training.
    This allows us to log validation metrics to Tensorboard.
    """
    
    def __init__(self, eval_period, evaluator, data_loader, tb_logger):
        """
        Initialize the hook.
        
        Args:
            eval_period (int): Evaluation frequency (in iterations)
            evaluator: Evaluator object
            data_loader: Data loader for validation set
            tb_logger: Tensorboard logger
        """
        self._eval_period = eval_period
        self._evaluator = evaluator
        self._data_loader = data_loader
        self._tb_logger = tb_logger
        self._best_ap = 0.0
    
    def _do_eval(self):
        """Run evaluation and log metrics."""
        results = self._evaluator.evaluate(self._data_loader)
        
        # Log metrics to Tensorboard
        if results and "bbox" in results:
            for k, v in results["bbox"].items():
                self._tb_logger.log_scalar(f"val/{k}", v, self.trainer.iter)
            
            # Save best model
            if results["bbox"]["AP"] > self._best_ap:
                self._best_ap = results["bbox"]["AP"]
                torch.save(
                    self.trainer.model.state_dict(),
                    os.path.join(self.trainer.cfg.OUTPUT_DIR, "model_best.pth")
                )
                print(f"New best model saved with AP: {self._best_ap:.4f}")
        
        return results
    
    def after_step(self):
        """Run after each training step."""
        if self.trainer.iter % self._eval_period == 0:
            self._do_eval()


class TensorboardTrainer(DefaultTrainer):
    """
    Trainer with Tensorboard logging.
    """
    
    def __init__(self, cfg, tb_logger):
        """
        Initialize the trainer.
        
        Args:
            cfg: Detectron2 config
            tb_logger: Tensorboard logger
        """
        super().__init__(cfg)
        self.tb_logger = tb_logger
        self.tb_event_writer = DetectronEventWriter(tb_logger)
    
    def build_hooks(self):
        """Build trainer hooks."""
        hooks = super().build_hooks()
        
        # 暂时禁用验证钩子，让训练能够正常启动
        """
        # Add validation hook
        eval_period = self.cfg.TEST.EVAL_PERIOD
        if eval_period > 0:
            data_loader = build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
            )
            evaluator = get_evaluator(
                self.cfg.DATASETS.TEST[0],
                self.cfg.OUTPUT_DIR,
            )
            hooks.insert(-1, ValidationHook(
                eval_period,
                evaluator,
                data_loader,
                self.tb_logger
            ))
        """
        print("训练期间暂时禁用了验证，将在训练完成后单独进行测试。")
        
        return hooks
    
    def after_step(self):
        """Run after each training step."""
        # 首先调用父类方法
        super().after_step()
        
        # 然后记录指标到Tensorboard
        with EventStorage(self.iter) as storage:
            for k, v in storage.latest().items():
                if "loss" in k:
                    self.tb_logger.log_scalar(f"train/{k}", v, self.iter)
    
    def after_train(self):
        """Run after training."""
        super().after_train()
        # Close Tensorboard logger
        if hasattr(self, "tb_logger"):
            self.tb_logger.close()


def setup_config(args):
    """
    Set up configuration for Sparse R-CNN.
    
    Args:
        args: Command-line arguments
        
    Returns:
        cfg: Detectron2 config
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/sparse_rcnn_R_50_FPN_100_proposals_3x.yaml"))
    
    # Set data directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voc_dir = os.path.join(root_dir, "data", "VOCdevkit")
    voc2007_dir = os.path.join(voc_dir, "VOC2007")
    
    # Print actual dataset paths for debugging
    print(f"VOC directory: {voc_dir}")
    print(f"VOC2007 directory: {voc2007_dir}")
    print(f"Main directory exists: {os.path.exists(os.path.join(voc2007_dir, 'ImageSets', 'Main'))}")
    
    # Manually set dataset paths for VOC
    cfg.DATASETS.TRAIN = ("voc_2007_train",)
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
        
        # 定义一个帮助函数，避免lambda闭包问题
        def get_voc_dicts(split_name, dirname=voc2007_dir, classes=voc_classes):
            return load_voc_instances(dirname, split_name, class_names=classes)
        
        # 直接注册每个数据集
        DatasetCatalog.register("voc_2007_train", lambda: get_voc_dicts("train"))
        MetadataCatalog.get("voc_2007_train").set(
            thing_classes=voc_classes, dirname=voc2007_dir, year=2007, split="train"
        )
        
        DatasetCatalog.register("voc_2007_val", lambda: get_voc_dicts("val"))
        MetadataCatalog.get("voc_2007_val").set(
            thing_classes=voc_classes, dirname=voc2007_dir, year=2007, split="val"
        )
        
        DatasetCatalog.register("voc_2007_test", lambda: get_voc_dicts("test"))
        MetadataCatalog.get("voc_2007_test").set(
            thing_classes=voc_classes, dirname=voc2007_dir, year=2007, split="test"
        )
        
        print("VOC datasets registered successfully")
    except Exception as e:
        print(f"Error registering VOC datasets: {e}")
    
    # Model parameters
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/sparse_rcnn_R_50_FPN_100_proposals_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # VOC has 20 classes
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (args.max_iter // 3, args.max_iter * 2 // 3)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    
    # Sparse R-CNN specific parameters
    cfg.MODEL.SPARSE_RCNN.NUM_PROPOSALS = 100
    cfg.MODEL.SPARSE_RCNN.NUM_CLASSES = 20
    
    # Testing parameters
    cfg.TEST.EVAL_PERIOD = args.eval_period
    
    # Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"sparse_rcnn_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    return cfg


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Sparse R-CNN on VOC dataset")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-iter", type=int, default=30000, help="Maximum iterations")
    parser.add_argument("--checkpoint-period", type=int, default=5000, help="Checkpoint period")
    parser.add_argument("--eval-period", type=int, default=1000, help="Evaluation period")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up configuration
    cfg = setup_config(args)
    
    # Set up Tensorboard logger
    tb_logger = setup_tensorboard(cfg.OUTPUT_DIR, "sparse_rcnn")
    
    # Create trainer
    trainer = TensorboardTrainer(cfg, tb_logger)
    
    # Train
    print(f"Starting training with {cfg.MODEL.DEVICE}...")
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print(f"Training completed. Results saved to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main() 