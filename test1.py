from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import CenterNet as net
import os
from coco.coco import CenterNetTestConfig, CocoDataset
from utils import utils as cocoutils
from PIL import Image
from utils import visualize
from utils.utils import resize_mask
import matplotlib.pyplot as plt
import cv2


def ccc(config, num_select, index, select_bbox, exist_i, class_seg, num_i, pic_preg):
    if num_i == 0:
        masks = np.zeros([int(config.IMAGE_MAX_DIM//config.STRIDE), int(config.IMAGE_MAX_DIM//config.STRIDE), num_select], np.float32)
    else:
        masks = seg_instance(config, num_select, index, select_bbox, exist_i, class_seg, num_i, pic_preg)
    return masks


def seg_instance(config, num_select, index, select_bbox, exist_i, class_seg, num_i, pic_preg):
    # [num_g_of_i]
    box_i = select_bbox[exist_i, ...]
    plt.imshow(class_seg)
    plt.show()
    # shape of coordinate equals [h_y_num, w_x_mun]
    h = range(int(config.IMAGE_MAX_DIM//config.STRIDE))
    [meshgrid_x, meshgrid_y] = np.meshgrid(h, h)
    meshgrid_y = np.expand_dims(meshgrid_y, axis=-1) + 0.5  # [Y, X, -1]
    meshgrid_x = np.expand_dims(meshgrid_x, axis=-1) + 0.5

    box_y1 = np.reshape(box_i[..., 0], [1, 1, -1])
    box_x1 = np.reshape(box_i[..., 1], [1, 1, -1])
    box_y2 = np.reshape(box_i[..., 2], [1, 1, -1])
    box_x2 = np.reshape(box_i[..., 3], [1, 1, -1])
    dist_l = meshgrid_x - box_x1  # (y, x, num_g)
    dist_r = box_x2 - meshgrid_x
    dist_t = meshgrid_y - box_y1
    dist_b = box_y2 - meshgrid_y

    # [y, x, num]
    grid_y_mask = (dist_t > 0.).astype(np.float32) * (dist_b > 0.).astype(np.float32)
    grid_x_mask = (dist_l > 0.).astype(np.float32) * (dist_r > 0.).astype(np.float32)

    class_seg = np.expand_dims(class_seg, axis=-1)
    n_seg = np.tile(class_seg, [1, 1, num_i])  # (y, x, num_g)
    rect_mask = grid_y_mask * grid_x_mask
    # for i in range(num_i):
    #     plt.imshow(rect_mask[..., i])
    #     plt.show()

    dependent = np.expand_dims((np.sum(rect_mask, axis=2)>1).astype(np.float32), -1)
    common_mask_n = rect_mask * dependent * n_seg
    # for i in range(num_i):
    #     plt.imshow(common_mask_n[..., i])
    #     plt.show()

    seperate_mask = rect_mask * n_seg - common_mask_n * n_seg
    # for i in range(num_i):
    #     plt.imshow(seperate_mask[..., i])
    #     plt.show()

    p_x1 = meshgrid_x - np.expand_dims(pic_preg[..., 0], -1)
    p_x2 = meshgrid_x + np.expand_dims(pic_preg[..., 1], -1)
    p_y1 = meshgrid_y - np.expand_dims(pic_preg[..., 2], -1)
    p_y2 = meshgrid_y + np.expand_dims(pic_preg[..., 3], -1)

    inter_width = np.minimum(box_x2, p_x2) - np.maximum(box_x1, p_x1)
    inter_height = np.minimum(box_y2, p_y2) - np.maximum(box_y1, p_y1)
    inter_area = inter_width * inter_height
    union_area = (box_y2 - box_y1) * (box_x2 - box_x1) + (p_y2 - p_y1) * (p_x2 - p_x1) - inter_area
    iou = inter_area / (union_area + 1e-12)

    iou_mask = iou * common_mask_n
    # for i in range(num_i):
    #     plt.imshow(iou_mask[..., i])
    #     plt.show()
    iou_max = np.expand_dims(np.amax(iou_mask, axis=2), -1)

    divide_mask = (np.equal(iou_mask, iou_max)).astype(np.float32)*common_mask_n
    masks = seperate_mask + divide_mask
    mask_score = masks*iou
    # for i in range(num_i):
    #     plt.imshow(mask_score[..., i])
    #     plt.show()
    masks = (mask_score > config.BOX_THRESHOLD).astype(np.float32)
    temp_masks = np.zeros([int(config.IMAGE_MAX_DIM//config.STRIDE), int(config.IMAGE_MAX_DIM//config.STRIDE), num_select], np.float32)
    temp_masks[..., index] = masks
    return temp_masks

# Set Coco
ROOT_DIR = os.path.abspath("../")

Config = CenterNetTestConfig()
Config.display()

# dataset = CocoDataset()
# COCO_DIR = ROOT_DIR + "/coco2014"
# dataset.load_coco(Config, COCO_DIR, "train", class_ids=[1, 17, 18])
# dataset.prepare()
class_names = {0:"BG",1:"person",2:"bicycle",3:"car",4:"motorcycle",5:"airplane",6:"bus",7:"train",8:"truck",9:"boat",10:"traffic light",11:"fire hydrant",
               12:"stop sign",13:"parking meter",14:"bench",15:"bird",16:"cat",17:"dog",18:"horse",19:"sheep",20:"cow",21:"elephant",22:"bear",
               23:"zebra",24:"giraffe",25:"backpack",26:"umbrella",27:"handbag",28:"tie",29:"suitcase",30:"frisbee",31:"skis",32:"snowboard",
               33:"sports ball",34:"kite",35:"baseball bat",36:"baseball glove",37:"skateboard",38:"surfboard",39:"tennis racket",40:"bottle",
               41:"wine glass",42:"cup",43:"fork",44:"knife",45:"spoon",46:"bowl",47:"banana",48:"apple",49:"sandwich",50:"orange",51:"broccoli",
               52:"carrot",53:"hot dog",54:"pizza",55:"donut",56:"cake",57:"chair",58:"couch",59:"potted plant",60:"bed",61:"dining table",
               62:"toilet",63:"tv",64:"laptop",65:"mouse",66:"remote",67:"keyboard",68:"cell phone",69:"microwave",70:"oven",71:"toaster",
               72:"sink",73:"refrigerator",74:"book",75:"clock",76:"vase",77:"scissors",78:"teddy bear",79:"hair drier",80:"toothbrush"}


centernet = net.CenterNet(Config, "pc")

# centernet.train_epochs(dataset, valset, Config, 50)
image = Image.open('COCO_val2014_000000000761.jpg')
#image = Image.open('COCO_val2014_000000000241.jpg')
# image = Image.open('COCO_train2014_000000000113.jpg')
# image = Image.open('COCO_train2014_000000005083.jpg')
# image = Image.open('COCO_val2014_000000000474.jpg')
#image = Image.open('COCO_val2014_000000034372.jpg')
# image = Image.open('people.jpeg')
# image = Image.open('dogs.jpg')
# image = Image.open('children.jpg')
# image = Image.open('catdog.jpeg')
# image = Image.open('car3.jpeg')
# image = Image.open('COCO_val2014_000000005535.jpg')


image = np.array(image)
image, window, scale, padding, crop = cocoutils.resize_image(
            image,
            min_dim=Config.IMAGE_MIN_DIM,
            min_scale=Config.IMAGE_MIN_SCALE,
            max_dim=Config.IMAGE_MAX_DIM,
            mode=Config.IMAGE_RESIZE_MODE)

[select_center, select_scores, select_bbox, select_class_id, class_seg, pic_preg] = centernet.test_one_image(image)

select_center = select_center[0]
select_class_id = select_class_id[0]
select_scores = select_scores[0]
select_bbox = select_bbox[0]

class_seg = class_seg[0]

# plt.imshow(class_seg[:, :, 0])
# plt.show()
pic_preg = pic_preg[0]
print(np.shape(select_center))
print(np.shape(select_class_id))
print(np.shape(select_scores))
print(np.shape(select_bbox))

print(np.shape(class_seg))
print(np.shape(pic_preg))


if np.shape(select_center)[0] > Config.DETECTION_MAX_INSTANCES:
    num_select = Config.DETECTION_MAX_INSTANCES
else:
    num_select = np.shape(select_center)[0]
select_scores = select_scores[0:num_select, ...]
select_center = select_center[0:num_select, ...]
select_class_id = select_class_id[0:num_select, ...]
select_bbox = select_bbox[0:num_select, ...]

# plt.imshow(class_seg[..., 1])
# plt.show()

final_masks = np.zeros([int(Config.IMAGE_MAX_DIM//Config.STRIDE), int(Config.IMAGE_MAX_DIM//Config.STRIDE), num_select], np.float32)

for i in range(Config.NUM_CLASSES):
    exist_i = np.equal(select_class_id, i)  # [0,1,...]
    exist_int = exist_i.astype(int)
    index = np.where(exist_int>0)[0]  # [a, b, 5, 8..]
    num_i = np.sum(exist_int)
    masks = ccc(Config, num_select, index, select_bbox, exist_i, class_seg[..., i], num_i, pic_preg)
    final_masks = final_masks + masks

# TODO: resize masks
padding = [(0, 0), (0, 0), (0, 0)]
stride_mask = resize_mask(final_masks, Config.STRIDE, padding, 0)
stride_mask = cv2.medianBlur(stride_mask, 5)
masks = stride_mask.astype(np.uint8).astype(np.float)
if len(np.shape(masks)) is 2:
    masks = np.expand_dims(masks, -1)
class_names = {0:"bg", 1:'person', 2:"car"}
visualize.display_instances(image, select_center*8+3, select_bbox*8, masks, select_class_id + 1, class_names, select_scores, show_mask=True)







