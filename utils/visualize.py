from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

PALETTE = [[ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
            [ 0, 153, 153 ],[ 0, 128, 255 ]]

def visualize(mask):
    h, w = mask.shape
    c = 3
    out = np.zeros((h, w, c))
    for i in range(len(PALETTE)):
        out[mask==i] = PALETTE[i]

    print(torch.Tensor(out).unique(return_counts=True))
    
    # plt.imshow(out / 255)
    # plt.show()
    return out / 255

def main():
    filename = "creek_00001"
    label_path = "/home/ztan4/Documents/TRSC/output/RUGD/masks/{}.png".format(filename)
    label = torch.tensor(np.array(Image.open(str(label_path))))
    visualize(label)

if __name__ == "__main__":
    main()