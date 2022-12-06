import numpy as np
import torch

def rgb2mask(img, color_id):
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

def convert_mask(mask, classes, inverse):
    temp = mask.numpy()
    if not inverse:
        label = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)    
        for v, k in classes.items():
            label[(temp == k["color"]).all(axis=2)] = 12 * v
    else:
        label = np.zeros(temp.shape + (3,), dtype=np.uint8)
        for v, k in classes.items():
            label[temp == v * 12, :] = k["color"]
    return torch.Tensor(label)