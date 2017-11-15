import sys
sys.path.insert(0, '../Mask_RCNN')

import os
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

import buildings

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights on COCO file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == "__main__":
    config = buildings.BuildingsConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    model_path = "models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)
    print("Weights successfully loaded")

    print("Preparing datasets")
    # Training dataset
    dataset_train = buildings.BuildingsDataset()
    dataset_train.load_buildings(dataset_dir="data/processed", subset="train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = buildings.BuildingsDataset()
    dataset_val.load_buildings(dataset_dir="data/processed", subset="val")
    dataset_val.prepare()

    # This training schedule is an example. Update to fit your needs.

    # Training - Stage 1
    # Adjust epochs and layers as needed
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Training Resnet layer 4+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=100,
                layers='4+')

    # Training - Stage 3
    # Finetune layers from ResNet stage 3 and up
    print("Training Resnet layer 3+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=200,
                layers='all')
