"""
CenterNetV2CenterNetV2
Configurations and data loading code for CenterNetV2.

Reference from:
<Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla>
and
<pycocotools https://github.com/cocodataset/cocoapi>

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import warnings
import random
import logging
import imgaug

sys.path.append("..")
from config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from utils import utils as cocoutils
from utils.utils import resize_mask

import matplotlib.pyplot as plt
import cv2


import zipfile
import urllib.request
import shutil
from scipy.sparse import csr_matrix

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import CenterNet2 config
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "centernetv2_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################


class CenterNetCocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if y
    # ou use a smaller GPU.
    IMAGES_PER_GPU = 6
    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 2  # COCO has 80 classes
    AUGMENT = False
    AUGMENTATION = None

    MODEL = 'train'
    GT_CHANNEL = 4+1+NUM_CLASSES*3
    L2_DECAY = 1e-4
    TRAIN_BN = True
    # PIC_NUM = 24
    PIC_NUM = 47700

    DETECTION_MAX_INSTANCES = 2500
    DETECTION_NMS_THRESHOLD = 0.3

    SEG_THRESHOLD = 0.5
    BOX_THRESHOLD = 0.2
    SCORE_THRESHOLD = 0.05

    VALIDATION_STEPS = 1


class CenterNetTestConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"
    AUGMENT = False
    AUGMENTATION = None

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16
    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1
    # Number of classes (including background)
    NUM_CLASSES = 2  # COCO has 80 classes

    MODEL = 'infer'
    GT_CHANNEL = 7
    L2_DECAY = 1e-4
    TRAIN_BN = False

    DETECTION_MAX_INSTANCES = 50
    DETECTION_NMS_THRESHOLD = 0.5

    SCORE_THRESHOLD = 0.3
    SEG_THRESHOLD = 0.7
    BOX_THRESHOLD = 0.1

############################################################
#  Dataset
############################################################

class CocoDataset(cocoutils.Dataset):
    # super_class properties:
    # _image_ids []
    # image_info [{source, image_id, path, width, height, annotations}, ...]
    # class_info = [{"source": "", "id": 0, "name": "BG"}, ...]
    # source_class_ids = {}
    def load_coco(self, config, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        self.config = config
        self.augment = self.config.AUGMENT
        self.augmentation = self.config.AUGMENTATION
        self.num_classes = self.config.NUM_CLASSES

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates, some class_id are contented in same image.
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                # add center of gravity???
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def prepare_image(self, image_id, augment=False, augmentation=None):
        """use config to processing coco image size and others,
        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.

        Returns:
        image: [height, width, 3]
        image_meta: the original shape of the image and resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image.
        gt_y: [instance_count]
        gt_x: [instance_count]
        vector_mask: [height, width, 2*class_num]. Set pixel relative center vector.
        """
        # Load image and mask
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        # original_shape = image.shape
        # print(original_shape)
        # print(type(original_shape))
        image, window, scale, padding, crop = cocoutils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE)
        mask = cocoutils.resize_mask(mask, scale, padding, 0, crop)

        # Random horizontal flips.
        # TODO: will be removed in a future update in favor of augmentation
        if self.augment:
            logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
            if random.randint(0, 1):
                image = np.fliplr(image)
                mask = np.fliplr(mask)

        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if self.augmentation:
            import imgaug

            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask.astype(np.uint8),
                                     hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            mask = mask.astype(np.bool)

        _idx = np.sum(mask, axis=(0, 1)) > 256
        # print(_idx)
        class_ids = class_ids[_idx]
        if len(class_ids) != 0:
            # print(class_ids)
            # [y, x, num_instance]
            mask = mask[:, :, _idx]
            # print(np.amax(mask, axis=(0, 1)))
            # Bounding boxes. Note that some boxes might be all zeros
            # if the corresponding mask got cropped out.
            # bbox: [num_instances, (y1, x1, y2, x2)]
            bbox = cocoutils.extract_bboxes(mask)
            gt_cy, gt_cx = cocoutils.gravity_center(mask)

            # Image meta data
            # image_meta = cocoutils.compose_image_meta(image_id, original_shape, image.shape, window, scale)
            # vector_mask = self.vector_mask(self.num_classes, class_ids, mask, gt_cx, gt_cy, image.shape, bbox)
            return image, class_ids, bbox, gt_cy, gt_cx, mask
        return None

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load class_vector masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a float [height, width, instances_count].

        Returns:
        masks: A  array of shape [height, width, instances_count] with
            one mask per class.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? iscrowd == 1, used RLE format, If so, use a negative class ID.
                if annotation['iscrowd']:
                    continue
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            # [height, width, instances_count]
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            # [instances_count]
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def generator(self, image_id):
        """Load and return ground truth data for an image (image, mask, bounding boxes).

        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.

        Returns:
        image: [height, width, 3]
        # shape: the original shape of the image before resizing and cropping.
        gt: {   class_ids: [instance_count] Integer class IDs,
                bbox: [instance_count, (y1, x1, y2, x2)]
            }
        stride_mask: [height, width, class_num*2]. The height and width are 1/4 those
            of the image.
        """
        # print("=========prepare for gt=========")
        gt = self.prepare_image(image_id, augment=self.augment, augmentation=self.augmentation)
        if gt is None:
            return None
        else:
            image, class_ids, bbox, gt_y, gt_x, gt_mask = gt

            floor_y = np.floor(gt_y / int(self.config.STRIDE)).astype(int)
            floor_x = np.floor(gt_x / int(self.config.STRIDE)).astype(int)

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            mean = np.reshape(mean, [1, 1, 3])
            std = np.reshape(std, [1, 1, 3])
            image = (image / 255. - mean) / std
            # print(class_ids)
            # bbox: [num_instances, (y1, x1, y2, x2)]
            gt_top = np.expand_dims((gt_y - bbox[..., 0]), axis=-1)
            gt_bot = np.expand_dims((bbox[..., 2] - gt_y), axis=-1)
            gt_left = np.expand_dims((gt_x - bbox[..., 1]), axis=-1)
            gt_right = np.expand_dims((bbox[..., 3] - gt_x), axis=-1)
            gt_y = np.expand_dims(gt_y, axis=-1)
            gt_x = np.expand_dims(gt_x, axis=-1)
            class_ids = np.expand_dims(class_ids, axis=-1)
            gt_basic = [gt_y, gt_x, gt_top, gt_left, gt_bot, gt_right, class_ids]
            gt = np.concatenate(gt_basic, axis=-1)
            instance_num = np.shape(gt)[0]

            # for normal distribution scalar
            scalar_y = []
            scalar_x = []
            # TODO: pshape without 56
            for i in range(instance_num):
                vertical_ones = np.tile([[1]], [self.config.IMAGE_MAX_DIM // int(self.config.STRIDE), 1])
                # print(np.shape(vertical_ones))
                horizontal_ones = tf.tile([[1]], [1, self.config.IMAGE_MAX_DIM // int(self.config.STRIDE)])
                pad_l = floor_x[i]
                pad_r = self.config.IMAGE_MAX_DIM // int(self.config.STRIDE) - floor_x[i] - 1
                vertical_pads = np.pad(vertical_ones, [[0, 0], [pad_l, pad_r]], "constant", constant_values=(
                int(gt_left[i][0] / self.config.STRIDE), int(gt_right[i][0] / self.config.STRIDE)))
                vertical_pads = np.expand_dims(vertical_pads, -1)
                # print(np.shape(vertical_pads))
                pad_t = floor_y[i]
                pad_b = self.config.IMAGE_MAX_DIM // int(self.config.STRIDE) - floor_y[i] - 1
                horizontal_pads = np.pad(horizontal_ones, [[pad_t, pad_b], [0, 0]], "constant", constant_values=(
                int(gt_top[i][0] / self.config.STRIDE), int(gt_bot[i][0] / self.config.STRIDE)))
                horizontal_pads = np.expand_dims(horizontal_pads, -1)
                # print(np.shape(horizontal_pads))
                scalar_y.append(horizontal_pads)
                scalar_x.append(vertical_pads)
            scalar_y = np.concatenate(scalar_y, axis=-1)  # [y, x, num_g]
            scalar_x = np.concatenate(scalar_x, axis=-1)  # [y, x, num_g]
            # print(np.shape(scalar_x))
            padding = [(0, 0), (0, 0), (0, 0)]
            stride_mask = resize_mask(gt_mask, 1 / self.config.STRIDE, padding, 0)
            masks = stride_mask.astype(np.uint8).astype(np.float)

            gravity_y = gt_y[..., 0] / self.config.STRIDE  # [gt_valid,1]
            gravity_x = gt_x[..., 0] / self.config.STRIDE
            gbbox_y1 = bbox[..., 0] / self.config.STRIDE  # not a rect-box shape, is mask shape
            gbbox_x1 = bbox[..., 1] / self.config.STRIDE
            gbbox_y2 = bbox[..., 2] / self.config.STRIDE
            gbbox_x2 = bbox[..., 3] / self.config.STRIDE
            class_id = class_ids[..., 0]

            # for gravity center
            gravi_yx = gt[..., 0:2] / self.config.STRIDE
            gravi_yx_round_int = np.floor(gravi_yx).astype(int)

            gravity_y = np.reshape(gravity_y, [1, 1, -1])
            gravity_x = np.reshape(gravity_x, [1, 1, -1])

            # for mask part
            gravity_y_tile = np.reshape(gravity_y, [1, 1, -1])
            gravity_x_tile = np.reshape(gravity_x, [1, 1, -1])

            gbbox_y1 = np.reshape(gbbox_y1, [1, 1, -1])
            gbbox_x1 = np.reshape(gbbox_x1, [1, 1, -1])
            gbbox_y2 = np.reshape(gbbox_y2, [1, 1, -1])
            gbbox_x2 = np.reshape(gbbox_x2, [1, 1, -1])
            num_g = np.shape(gbbox_y1)[-1]

            h = np.arange(self.config.IMAGE_MAX_DIM / self.config.STRIDE).astype(np.float32)
            w = np.arange(self.config.IMAGE_MAX_DIM / self.config.STRIDE).astype(np.float32)

            # shape of coordinate equals [h_y_num, w_x_mun]
            [grid_x, grid_y] = tf.meshgrid(w, h)

            grid_y = np.expand_dims(grid_y, -1)
            grid_x = np.expand_dims(grid_x, -1)
            grid_y = np.tile(grid_y, [1, 1, num_g])  # (y, x, num_g)
            grid_x = np.tile(grid_x, [1, 1, num_g])
            dist_l = grid_x - gbbox_x1  # (y, x, num_g)
            dist_r = gbbox_x2 - grid_x
            dist_t = grid_y - gbbox_y1
            dist_b = gbbox_y2 - grid_y
            grid_y_mask = (dist_t > 0.).astype(np.float32) * (dist_b > 0.).astype(np.float32)
            grid_x_mask = (dist_l > 0.).astype(np.float32) * (dist_r > 0.).astype(np.float32)

            heatmask = grid_y_mask * grid_x_mask * masks  # not a rect-box shape, is mask shape (y, x, num_g)
            dist_l *= heatmask  # not a rect-box shape, is mask shape shape (y, x, num_g)
            dist_r *= heatmask
            dist_t *= heatmask
            dist_b *= heatmask

            loc = np.max(heatmask, axis=-1)  # (y, x) objects mask
            dist_area = (dist_l + dist_r) * (dist_t + dist_b)  # not a rect-box shape, is mask shape shape (y, x, num_g)
            dist_area_ = dist_area + (1. - heatmask) * 1e8
            dist_area_min = np.min(dist_area_, axis=-1, keepdims=True)  # small things on the top, background is 1e8
            # not overlap things mask (y, x, num_g)
            dist_mask = np.equal(dist_area, dist_area_min).astype(np.float32) * np.expand_dims(loc, axis=-1)

            # gravity_y_tile = dist_mask * gravity_y_tile  # not a rect-box shape, gravity center mask (y, x, num_g)
            # gravity_x_tile = dist_mask * gravity_x_tile
            # gbbox_y1 *= dist_mask  # (y, x, num_g)
            # gbbox_x1 *= dist_mask
            # gbbox_y2 *= dist_mask
            # gbbox_x2 *= dist_mask

            dist_l *= dist_mask  # valid dist l, r, t, b
            dist_r *= dist_mask
            dist_t *= dist_mask
            dist_b *= dist_mask
            dist_l = np.expand_dims(np.max(dist_l, axis=-1), -1)  # not overlap 1 (y, x)
            dist_r = np.expand_dims(np.max(dist_r, axis=-1), -1)
            dist_t = np.expand_dims(np.max(dist_t, axis=-1), -1)
            dist_b = np.expand_dims(np.max(dist_b, axis=-1), -1)
            gt_reg = np.concatenate([dist_t, dist_l, dist_b, dist_r], axis=-1).astype(np.float32)

            # for normal distribution
            reduction = np.exp(-(((grid_y - gravity_y // 1) / np.sqrt(scalar_y + 1e-8)) ** 2 +
                                 ((grid_x - gravity_x // 1) / np.sqrt(scalar_x + 1e-8)) ** 2) / (2 * 1 ** 2))
            iou_reduction = np.max(reduction, axis=2)  # [y, x, num_g] --> [y, x]
            iou_reduction = np.expand_dims(loc * iou_reduction, axis=-1).astype(np.float32)
            # iou_reduction = np.expand_dims(loc, axis=-1).astype(np.float32)

            zero_like = np.zeros((int(self.config.IMAGE_MAX_DIM / self.config.STRIDE),
                                  int(self.config.IMAGE_MAX_DIM / self.config.STRIDE), 1), np.float32)
            gt_keypoints = []
            heatmap_gt = []
            reduction_gt = []
            for i in range(self.num_classes):
                # [num_g, 1]
                exist_i = np.equal(class_id - 1, i)  # pass BG CLASS_ID: 0
                gy = gravi_yx_round_int[..., 0][exist_i]
                gx = gravi_yx_round_int[..., 1][exist_i]
                num_i = np.sum(exist_i.astype(np.int32))
                # [num_g_of_i, y, x]
                reduce_i = reduction[..., exist_i]
                heatmask_i = dist_mask[..., exist_i]
                gbbox_yx_i = gravi_yx_round_int[exist_i, ...]
                # [y, x, 1] heat_map for class i , if null class i, product zero_like_map
                if np.shape(reduce_i)[-1] == 0:
                    reduce_i = zero_like
                    heatmap_i = zero_like
                    gt_keypoints_i = zero_like
                else:
                    reduce_i = np.expand_dims(np.max(reduce_i, axis=2), axis=-1)
                    heatmap_i = np.expand_dims(np.max(heatmask_i, axis=2), axis=-1)
                    gt_keypoints_i = csr_matrix((np.ones(num_i), (gy, gx)),
                                                shape=(int(self.config.IMAGE_MAX_DIM / self.config.STRIDE),
                                                       int(self.config.IMAGE_MAX_DIM / self.config.STRIDE))).toarray()
                    gt_keypoints_i = np.expand_dims(gt_keypoints_i, -1)
                reduction_gt.append(reduce_i)
                heatmap_gt.append(heatmap_i)
                gt_keypoints.append(gt_keypoints_i)
            reduction_gt = np.concatenate(reduction_gt, axis=-1).astype(np.float32)
            heatmap_gt = np.concatenate(heatmap_gt, axis=-1).astype(np.float32)
            gt_keypoints = np.concatenate(gt_keypoints, axis=-1).astype(np.float32)

            # plt.imshow(image)
            # plt.show()
            # plt.imshow(iou_reduction[:, :, 0])
            # plt.show()
            # plt.imshow(dist_l[:,:,0])
            # plt.show()
            # plt.imshow(gt_keypoints[:, :, 0])
            # plt.show()

            gt = np.concatenate([gt_reg, iou_reduction, gt_keypoints, reduction_gt, heatmap_gt], axis=-1)

            return image, gt

class SequenceData(tf.keras.utils.Sequence):
    def __init__(self, num_imgs, batch_size, dataset, img_idx):
        # 初始化所需的参数
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_imgs = num_imgs
        self.img_idx = img_idx
        print("=============sequencdData==========")

    def __len__(self):
        return np.math.ceil(self.num_imgs / self.batch_size)

    def __getitem__(self, idx):
        # 迭代器部分
        imgs = []
        gt_batch = []
        b = 0
        id = idx * self.batch_size
        while True:
            if self.img_idx >= self.num_imgs:
                self.img_idx = 0
            if id >= self.num_imgs:
                id = 0
            if b >= self.batch_size:
                # print("==")
                inputs = (imgs, gt_batch)
                targets = []
                return inputs, targets
            res = self.dataset.generator(image_id=id)
            if res is not None:
                img, gt = res
                imgs.append(img)
                gt_batch.append(gt)
                b += 1
            id += 1
            self.img_idx += 1


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True,
                                     auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
