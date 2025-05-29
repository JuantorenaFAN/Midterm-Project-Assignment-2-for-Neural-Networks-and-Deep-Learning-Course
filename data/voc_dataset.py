#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import json
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
import xml.etree.ElementTree as ET
from PIL import Image

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VOC_DIR = os.path.join(DATA_DIR, "VOCdevkit")


def get_voc_dicts(split, year="2007"):
    """
    Load and process VOC dataset annotations into Detectron2 format.
    
    Args:
        split (str): "train", "val", "trainval", or "test"
        year (str): "2007" or "2012"
        
    Returns:
        list: List of dataset dictionaries
    """
    if year == "2007+2012" and split == "trainval":
        # Combine 2007 and 2012 trainval
        dataset_dicts = get_voc_dicts("trainval", "2007") + get_voc_dicts("trainval", "2012")
        return dataset_dicts
    
    data_dir = os.path.join(VOC_DIR, f"VOC{year}")
    annotation_dir = os.path.join(data_dir, "Annotations")
    image_dir = os.path.join(data_dir, "JPEGImages")
    split_file = os.path.join(data_dir, "ImageSets", "Main", f"{split}.txt")
    
    with open(split_file, "r") as f:
        fileids = [line.strip() for line in f.readlines()]
    
    dataset_dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dir, f"{fileid}.xml")
        tree = ET.parse(anno_file)
        root = tree.getroot()
        
        image_file = os.path.join(image_dir, f"{fileid}.jpg")
        
        # Extract image dimensions
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        record = {
            "file_name": image_file,
            "image_id": fileid,
            "height": height,
            "width": width,
        }
        
        # Extract annotations
        objs = []
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in VOC_CLASSES:
                continue
            
            # VOC dataset contains difficult flag
            difficult = int(obj.find("difficult").text)
            
            bbox = obj.find("bndbox")
            box = [
                float(bbox.find("xmin").text),
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
            ]
            
            obj_dict = {
                "bbox": box,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": VOC_CLASSES.index(cls),
                "iscrowd": 0,
                "difficult": difficult,
            }
            objs.append(obj_dict)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


def register_voc_datasets():
    """Register VOC datasets to Detectron2."""
    # Register VOC2007 train, val, and test
    for split in ["train", "val", "test", "trainval"]:
        name = f"voc_2007_{split}"
        # Check if dataset is already registered
        if name in DatasetCatalog:
            print(f"Dataset '{name}' is already registered, skipping registration.")
            continue
        DatasetCatalog.register(name, lambda s=split: get_voc_dicts(s, "2007"))
        MetadataCatalog.get(name).set(
            thing_classes=list(VOC_CLASSES),
            year=2007,
            split=split,
            evaluator_type="pascal_voc",
        )
    
    # Register VOC2012 train and val
    for split in ["train", "val", "trainval"]:
        name = f"voc_2012_{split}"
        # Check if dataset is already registered
        if name in DatasetCatalog:
            print(f"Dataset '{name}' is already registered, skipping registration.")
            continue
        DatasetCatalog.register(name, lambda s=split: get_voc_dicts(s, "2012"))
        MetadataCatalog.get(name).set(
            thing_classes=list(VOC_CLASSES),
            year=2012,
            split=split,
            evaluator_type="pascal_voc",
        )
    
    # Register combined VOC2007+2012 trainval
    name = "voc_2007+2012_trainval"
    if name in DatasetCatalog:
        print(f"Dataset '{name}' is already registered, skipping registration.")
    else:
        DatasetCatalog.register(name, lambda: get_voc_dicts("trainval", "2007+2012"))
        MetadataCatalog.get(name).set(
            thing_classes=list(VOC_CLASSES),
            evaluator_type="pascal_voc",
        )


def convert_voc_to_coco_format(split, year="2007", output_json=None):
    """
    Convert VOC annotations to COCO format and save as JSON.
    This is useful for models that expect COCO format.
    
    Args:
        split (str): "train", "val", "trainval", or "test"
        year (str): "2007" or "2012"
        output_json (str): Path to output JSON file
    
    Returns:
        dict: COCO format dataset
    """
    voc_dicts = get_voc_dicts(split, year)
    
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(VOC_CLASSES)]
    }
    
    ann_id = 1
    for image_id, record in enumerate(voc_dicts):
        # Image info
        coco_dict["images"].append({
            "id": image_id,
            "file_name": os.path.basename(record["file_name"]),
            "height": record["height"],
            "width": record["width"],
        })
        
        # Annotations
        for annotation in record["annotations"]:
            bbox = annotation["bbox"]
            # COCO format uses [x, y, width, height]
            coco_bbox = [
                bbox[0],  # x
                bbox[1],  # y
                bbox[2] - bbox[0],  # width
                bbox[3] - bbox[1],  # height
            ]
            
            # Get segmentation from box (simplified)
            x1, y1, x2, y2 = bbox
            segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            
            area = coco_bbox[2] * coco_bbox[3]
            
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": annotation["category_id"],
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": annotation["iscrowd"],
                "segmentation": segmentation,
            })
            
            ann_id += 1
    
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(coco_dict, f)
    
    return coco_dict


def prepare_voc_datasets():
    """Prepare VOC datasets for training and evaluation."""
    print("Registering VOC datasets")
    register_voc_datasets()
    
    # Also convert to COCO format for some models
    coco_dir = os.path.join(DATA_DIR, "coco_format")
    os.makedirs(coco_dir, exist_ok=True)
    
    datasets = [
        ("voc_2007_train", "2007", "train"),
        ("voc_2007_val", "2007", "val"),
        ("voc_2007_test", "2007", "test"),
        ("voc_2007+2012_trainval", "2007+2012", "trainval"),
    ]
    
    for name, year, split in datasets:
        output_json = os.path.join(coco_dir, f"{name}.json")
        if not os.path.exists(output_json):
            print(f"Converting {name} to COCO format")
            convert_voc_to_coco_format(split, year, output_json)
    
    # Register COCO format datasets
    for name, _, _ in datasets:
        coco_name = f"{name}_coco"
        # Check if COCO format dataset is already registered
        if coco_name in DatasetCatalog:
            print(f"COCO format dataset '{coco_name}' is already registered, skipping registration.")
            continue
            
        json_file = os.path.join(coco_dir, f"{name}.json")
        image_root = os.path.join(VOC_DIR, "VOC2007", "JPEGImages")
        register_coco_instances(coco_name, {}, json_file, image_root)
    
    print("VOC datasets prepared successfully")


if __name__ == "__main__":
    prepare_voc_datasets() 