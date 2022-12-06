# import pathlib
# import torch
# from torchvision.utils import save_image
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from utils.utils import *

# CLASS = {
#     0: {'name': 'dirt', 'color': [108, 64, 20]}, 
#     1: {'name': 'sand', 'color': [255, 229, 204]}, 
#     2: {'name': 'grass', 'color': [0, 102, 0]}, 
#     3: {'name': 'tree', 'color': [0, 255, 0]}, 
#     4: {'name': 'pole', 'color': [0, 153, 153]}, 
#     5: {'name': 'water', 'color': [0, 128, 255]},
#     6: {'name': 'sky', 'color': [0, 0, 255]}, 
#     7: {'name': 'vehicle', 'color': [255, 255, 0]}, 
#     8: {'name': 'container/generic-object', 'color': [255, 0, 127]}, 
#     9: {'name': 'asphalt', 'color': [64, 64, 64]}, 
#     10: {'name': 'gravel', 'color': [255, 128, 0]}, 
#     11: {'name': 'building', 'color': [255, 0, 0]}, 
#     12: {'name': 'mulch', 'color': [153, 76, 0]}, 
#     13: {'name': 'rock-bed', 'color': [102, 102, 0]}, 
#     14: {'name': 'log', 'color': [102, 0, 0]}, 
#     15: {'name': 'bicycle', 'color': [0, 255, 128]}, 
#     16: {'name': 'person', 'color': [204, 153, 255]}, 
#     17: {'name': 'fence', 'color': [102, 0, 204]}, 
#     18: {'name': 'bush', 'color': [255, 153, 204]}, 
#     19: {'name': 'sign', 'color': [0, 102, 102]}, 
#     20: {'name': 'rock', 'color': [153, 204, 255]}, 
#     21: {'name': 'bridge', 'color': [102, 255, 255]}, 
#     22: {'name': 'concrete', 'color': [101, 101, 11]}, 
#     23: {'name': 'picnic-table', 'color': [114, 85, 47]}
# }

# def main():
#     output_dir = "./output/RUGD/masks/{}.png"
#     color_masks_dir = pathlib.Path("./data/RUGD/masks")
#     color_masks = sorted(list(color_masks_dir.glob('*.png')))

#     count = 1
#     for color_mask in color_masks:
#         mask = torch.tensor(np.array(Image.open(str(color_mask))))
#         label = convert_mask(mask, False)

#         reverted_mask = convert_mask(label, True)
#         assert (reverted_mask.numpy() == mask.numpy()).all()
#         save_label = Image.fromarray(label.numpy())
#         # imageio.imwrite(output_dir.format(color_mask.stem), label)
#         # save_image(label.numpy(), output_dir.format(color_mask.stem))
        
#         print("{}: ".format(count), color_mask.stem, end='\r')
#         count += 1

# def convert_mask(mask, inverse):
#     temp = mask.numpy()
#     if not inverse:
#         label = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)    
#         for v, k in CLASS.items():
#             label[(temp == k["color"]).all(axis=2)] = 12 * v
#     else:
#         label = np.zeros(temp.shape + (3,), dtype=np.uint8)
#         for v, k in CLASS.items():
#             label[temp == v * 12, :] = k["color"]
#     return torch.Tensor(label)

# if __name__ == "__main__":
#     main()

# Adapted from OFFSEG by Viswanath et al. 
import argparse
import os.path as osp
import numpy as np
import mmcv
# import cv2
from PIL import Image
import pathlib

rudg_dir = "./data/rugd/"
annotation_folder = "RUGD_annotations/"

CLASSES = ("dirt", "sand", "grass", "tree", "pole", "water", "sky", 
        "vehicle", "container/generic-object", "asphalt", "gravel", 
        "building", "mulch", "rock-bed", "log", "bicycle", "person", 
        "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table")

PALETTE = [ [ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
            [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
            [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
            [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ] ]

Groups = [2, 2, 2, 5, 5, 4, 0, 5, 5, 1, 2, 5, 2, 3, 5, 5, 5, 5, 5, 0, 3, 5, 1, 5]


# 0 -- Background: void, sky, sign
# 1 -- Level1 - Navigable: concrete, asphalt
# 2 -- Level2 - Navigable: gravel, grass, dirt, sand, mulch
# 3 -- Level3 - Navigable: Rock, Rock-bed
# 4 -- Non-Navigable: water
# 5 -- Obstacle:  tree, pole, vehicle, container/generic-object, building, log, 
#                 bicycle(could be removed), person, fence, bush, picnic-table, bridge,

color_id = {tuple(c):i for i, c in enumerate(PALETTE)}
color_id[tuple([0, 0, 0])] = 255

def rgb2mask(img):
    # assert len(img) == 3
    h, w, c = img.shape
    out = np.ones((h, w, c)) * 255
    for i in range(h):
        for j in range(w):
            if tuple(img[i, j]) in color_id:
                out[i][j] = color_id[tuple(img[i, j])]
            else:
                print("unknown color, exiting...")
                exit(0)
    return out


def raw_to_seq(seg):
    h, w = seg.shape
    out = np.zeros((h, w))
    for i in range(len(Groups)):
        out[seg==i] = Groups[i]

    out[seg==255] = 0
    return out


# with open(osp.join(rudg_dir, 'train_ours.txt'), 'r') as r:
mask_dir = pathlib.Path("./data/RUGD/masks")
color_masks = sorted(list(mask_dir.glob('*.png')))
out_dir = "./output/RUGD/masks/{}.png"
i = 0
for color_mask in color_masks:
    print("train: {}".format(i))
    # w.writelines(l[:-5] + "\n")
    # w.writelines(l.split(".")[0] + "\n")
    file_client_args=dict(backend='disk')
    file_client = mmcv.FileClient(**file_client_args)
    img_bytes = file_client.get(str(color_mask))

    gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
    # bgr to rgb
    gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
    out = rgb2mask(gt_semantic_seg)
    # a mask containing len(PALETTE) colors
    out = out[:, :, 0]

    # mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
    out2 = raw_to_seq(out)
    mmcv.imwrite(out2, out_dir.format(color_mask.stem))

    i += 1


# with open(osp.join(rudg_dir, 'val_ours.txt'), 'r') as r:
#     i = 0
#     for l in r:
#         print("val: {}".format(i))
#         # w.writelines(l[:-5] + "\n")
#         # w.writelines(l.split(".")[0] + "\n")
#         file_client_args=dict(backend='disk')
#         file_client = mmcv.FileClient(**file_client_args)
#         img_bytes = file_client.get(rudg_dir + annotation_folder + l.strip() + '.png')
#         gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
#         gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
#         out = rgb2mask(gt_semantic_seg)
#         out = out[:, :, 0]
#         mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
#         out2 = raw_to_seq(out)
#         mmcv.imwrite(out2, rudg_dir + annotation_folder + l.strip() + "_group6.png")

#         i += 1



# with open(osp.join(rudg_dir, 'test_ours.txt'), 'r') as r:
#     i = 0
#     for l in r:
#         print("test: {}".format(i))
#         # w.writelines(l[:-5] + "\n")
#         # w.writelines(l.split(".")[0] + "\n")
#         file_client_args=dict(backend='disk')
#         file_client = mmcv.FileClient(**file_client_args)
#         img_bytes = file_client.get(rudg_dir + annotation_folder + l.strip() + '.png')
#         gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
#         gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
#         out = rgb2mask(gt_semantic_seg)
#         out = out[:, :, 0]
#         mmcv.imwrite(out, rudg_dir + annotation_folder + l.strip()+ "_orig.png")
#         out2 = raw_to_seq(out)
#         mmcv.imwrite(out2, rudg_dir + annotation_folder + l.strip() + "_group6.png")

#         i += 1



print("successful")