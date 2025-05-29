#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.voc_dataset import prepare_voc_datasets

if __name__ == "__main__":
    print("Preparing VOC datasets...")
    prepare_voc_datasets()
    print("Datasets prepared successfully.") 