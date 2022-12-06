# import pathlib
# import torch
# from torchvision.utils import save_image
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from utils.utils import *

# CLASS = {
#     0: {"color": [0, 0, 0],  "name": "void"},
#     1: {"color": [108, 64, 20],   "name": "dirt"},
#     3: {"color": [0, 102, 0],   "name": "grass"},
#     4: {"color": [0, 255, 0],  "name": "tree"},
#     5: {"color": [0, 153, 153],  "name": "pole"},
#     6: {"color": [0, 128, 255],  "name": "water"},
#     7: {"color": [0, 0, 255],  "name": "sky"},
#     8: {"color": [255, 255, 0],  "name": "vehicle"},
#     9: {"color": [255, 0, 127],  "name": "object"},
#     10: {"color": [64, 64, 64],  "name": "asphalt"},
#     11: {"color": [255, 0, 0],  "name": "building"},
#     12: {"color": [102, 0, 0],  "name": "log"},
#     13: {"color": [204, 153, 255],  "name": "person"},
#     14: {"color": [102, 0, 204],  "name": "fence"},
#     15: {"color": [255, 153, 204],  "name": "bush"},
#     16: {"color": [170, 170, 170],  "name": "concrete"},
#     17: {"color": [41, 121, 255],  "name": "barrier"},
#     18: {"color": [134, 255, 239],  "name": "puddle"},
#     19: {"color": [99, 66, 34],  "name": "mud"},
#     20: {"color": [110, 22, 138],  "name": "rubble"}
# }
# ids = []
# def main():
#     output_dir = "./output/Rellis-3D/masks/{}.png"
#     color_masks_dir = pathlib.Path("./data/Rellis-3D/masks")
#     color_masks = sorted(list(color_masks_dir.glob('*.png')))

#     count = 1
#     for color_mask in color_masks:
#         mask = torch.tensor(np.array(Image.open(str(color_mask))))
#         label = convert_mask(mask, CLASS, False)

#         reverted_mask = convert_mask(label, CLASS, True)
#         assert (reverted_mask.numpy() == mask.numpy()).all()
#         save_label = Image.fromarray(label.numpy())
#         # imageio.imwrite(output_dir.format(color_mask.stem), label)
#         # save_image(label.numpy(), output_dir.format(color_mask.stem))
        
#         print("{}: ".format(count), color_mask.stem, end='\r')
#         count += 1



# if __name__ == "__main__":
#     # main()
    
#     ## testing1
#     # path = '/home/ztan4/Documents/TRSC/data/Rellis-3D/masks/frame000000-1581623790_349.png'
#     # mask = torch.tensor(np.array(Image.open(str(path))))
#     # label = convert_mask(mask, CLASS, False)
#     # print(label.shape)
#     # new_mask = convert_mask(label, CLASS, True)
#     # plt.imshow(new_mask / 255)
#     # plt.show()

#     ## testing2
#     filename = "frame000000-1581623790_349"
#     label_path = "/home/ztan4/Documents/TRSC/output/Rellis-3D/masks/{}.png".format(filename)
#     label = torch.tensor(np.array(Image.open(str(label_path))))
#     reverted_mask = convert_mask(label, CLASS, True)
#     mask_path = "./data/Rellis-3D/masks/{}.png".format(filename)
#     mask = torch.tensor(np.array(Image.open(str(mask_path))))
#     print((mask.numpy() == reverted_mask.numpy()))
#     print(torch.Tensor(label).unique())

# Adapted from OFFSEG by Viswanath et al. 
import argparse
import os.path as osp
import numpy as np
import mmcv
# import cv2
from PIL import Image
import pathlib

# 0 -- Background: void, sky, 
# 1 -- Level1 - Navigable: concrete, asphalt
# 2 -- Level2 - Navigable: dirt, grass, 
# 3 -- Level3 - Navigable: mud, rubble
# 4 -- Non-Navigable: water, bush, puddle,
# 5 -- Obstacle: tree, pole, vehicle, object, building, log, person, fence, barrier

CLASSES = ("void", "dirt", "grass", "tree", "pole", "water", "sky", "vehicle", 
            "object", "asphalt", "building", "log", "person", "fence", "bush", 
            "concrete", "barrier", "puddle", "mud", "rubble")

PALETTE = [[0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], 
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], 
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], 
            [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]

Groups = [0, 2, 2, 5, 5, 4, 0, 5, 5, 1, 5, 5, 5, 5, 4, 1, 5, 4, 3, 3]

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
mask_dir = pathlib.Path("./data/Rellis-3D/masks")
color_masks = sorted(list(mask_dir.glob('*.png')))
out_dir = "./output/Rellis-3D/masks/{}.png"
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