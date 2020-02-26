from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import numpy as np
import CenterNet as net
import os
from coco.coco import CenterNetCocoConfig, CocoDataset, SequenceData
from PIL import Image

# Set Coco
ROOT_DIR = os.path.abspath("../")
COCO_DIR = ROOT_DIR + "/coco2014"
Config = CenterNetCocoConfig()
Config.display()

dataset = CocoDataset()
# 1person 2bicycle 3car 4motorcycle 6bus 8truck 10traffic light 11fire hydrant 12street sign 13stop sign
dataset.load_coco(Config, COCO_DIR, "train", class_ids=[1, 3])
# dataset.load_coco(Config, COCO_DIR, "train", class_ids=[1, 2, 3, 4, 6, 8, 10, 11])
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# valset = CocoDataset()
# valset.load_coco(Config, COCO_DIR, "val", class_ids=[1, 17, 18])
# valset.prepare()
# print("Image Count: {}".format(len(valset.image_ids)))
# print("Class Count: {}".format(valset.num_classes))

# centernet = net.CenterNet(Config, "Creature")


centernet = net.CenterNet(Config, "pc")
seqdata = SequenceData(Config.PIC_NUM, Config.BATCH_SIZE, dataset, 0)
centernet.train_epochs(seqdata, None, Config, 50)

# for i in range(3):
#     dataset.generator(i)

