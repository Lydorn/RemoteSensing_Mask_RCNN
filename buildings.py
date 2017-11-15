"""
Mask R-CNN
Configurations and data loading code for the Buildings dataset.
"""

import sys
sys.path.insert(0, '../Mask_RCNN')

import os
import math
import random
import numpy as np
import skimage
import matplotlib
import matplotlib.pyplot as plt
import cv2

from config import Config
import utils

class BuildingsConfig(Config):
    """Configuration for training on the Buildings dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Give the configuration a recognizable name
    NAME = "buildings"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2  # background + building

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])  # TODO: change to real mean pixel


class BuildingsDataset(utils.Dataset):

    def load_buildings(self, dataset_dir="data/processed", subset="train"):
        """Load a subset of the Buildings dataset.
        dataset_dir: The root directory of the Buildings dataset.
        subset: What to load (train, val)
        """
        # Path
        image_dir = os.path.join(dataset_dir, subset, "images")
        gt_dir = os.path.join(dataset_dir, subset, "gt")

        # Add classes
        self.add_class("buildings", 1, "building")

        # Add images
        for i, file in enumerate(os.listdir(image_dir)):
            self.add_image(
                "buildings", image_id=i,
                path=os.path.join(image_dir, file),
                name=os.path.splitext(file)[0],
                mask_path=os.path.join(gt_dir, file))

    def load_mask(self, image_id):
        """"Load instance masks for the given image.

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        print("Loading masks for image {}".format(image_id))

        image_info = self.image_info[image_id]

        gt = skimage.io.imread(image_info['mask_path'])

        connectivity = 4
        output = cv2.connectedComponentsWithStats(gt, connectivity, cv2.CV_32S)
        num_instances = output[0]
        instances = output[1]

        instance_masks = []
        class_ids = []
        for i in range(num_instances):
            m = (instances == i + 1)
            if m.max() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(1)  # Only buildings

        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "buildings":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)


if __name__ == "__main__":
    print("Buildings")
