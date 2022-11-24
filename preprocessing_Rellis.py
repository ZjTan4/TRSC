import pathlib
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CLASS = {
    0: {"color": [0, 0, 0],  "name": "void"},
    1: {"color": [108, 64, 20],   "name": "dirt"},
    3: {"color": [0, 102, 0],   "name": "grass"},
    4: {"color": [0, 255, 0],  "name": "tree"},
    5: {"color": [0, 153, 153],  "name": "pole"},
    6: {"color": [0, 128, 255],  "name": "water"},
    7: {"color": [0, 0, 255],  "name": "sky"},
    8: {"color": [255, 255, 0],  "name": "vehicle"},
    9: {"color": [255, 0, 127],  "name": "object"},
    10: {"color": [64, 64, 64],  "name": "asphalt"},
    11: {"color": [255, 0, 0],  "name": "building"},
    12: {"color": [102, 0, 0],  "name": "log"},
    13: {"color": [204, 153, 255],  "name": "person"},
    14: {"color": [102, 0, 204],  "name": "fence"},
    15: {"color": [255, 153, 204],  "name": "bush"},
    16: {"color": [170, 170, 170],  "name": "concrete"},
    17: {"color": [41, 121, 255],  "name": "barrier"},
    18: {"color": [134, 255, 239],  "name": "puddle"},
    19: {"color": [99, 66, 34],  "name": "mud"},
    20: {"color": [110, 22, 138],  "name": "rubble"}
}
def main():
    output_dir = "./output/Rellis-3D/masks/{}.png"
    color_masks_dir = pathlib.Path("./data/Rellis-3D/masks")
    color_masks = sorted(list(color_masks_dir.glob('*.png')))

    count = 1
    for color_mask in color_masks:
        mask = torch.tensor(np.array(Image.open(str(color_mask))))
        label = convert_mask(mask, False)

        reverted_mask = convert_mask(label, True)
        assert (reverted_mask.numpy() == mask.numpy()).all()
        
        # imageio.imwrite(output_dir.format(color_mask.stem), label)
        # save_image(label.numpy(), output_dir.format(color_mask.stem))
        
        print("{}: ".format(count), color_mask.stem, end='\r')
        count += 1

def convert_mask(mask, inverse):
    temp = mask.numpy()
    if not inverse:
        label = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)    
        for v, k in CLASS.items():
            label[(temp == k["color"]).all(axis=2)] = 12 * v
    else:
        label = np.zeros(temp.shape + (3,), dtype=np.uint8)
        for v, k in CLASS.items():
            label[temp == v * 12, :] = k["color"]
    return torch.Tensor(label)

if __name__ == "__main__":
    # main()
    
    ## testing1
    # path = '/home/ztan4/Documents/TRSC/data/Rellis-3D/masks/frame000000-1581623790_349.png'
    # mask = torch.tensor(np.array(Image.open(str(path))))
    # label = convert_mask(mask, False)
    # print(label.shape)
    # new_mask = convert_mask(label, True)
    # plt.imshow(new_mask / 255)
    # plt.show()

    ## testing2
    filename = "frame000000-1581623790_349"
    label_path = "/home/ztan4/Documents/TRSC/output/Rellis-3D/masks/{}.png".format(filename)
    label = torch.tensor(np.array(Image.open(str(label_path))))
    reverted_mask = convert_mask(label, True)
    mask_path = "./data/Rellis-3D/masks/{}.png".format(filename)
    mask = torch.tensor(np.array(Image.open(str(mask_path))))
    print((mask.numpy() == reverted_mask.numpy()))
    print(torch.Tensor(label).unique())
