import pathlib
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
CLASS = {
    0: {'name': 'dirt', 'color': [108, 64, 20]}, 
    1: {'name': 'sand', 'color': [255, 229, 204]}, 
    2: {'name': 'grass', 'color': [0, 102, 0]}, 
    3: {'name': 'tree', 'color': [0, 255, 0]}, 
    4: {'name': 'pole', 'color': [0, 153, 153]}, 
    5: {'name': 'water', 'color': [0, 128, 255]},
    6: {'name': 'sky', 'color': [0, 0, 255]}, 
    7: {'name': 'vehicle', 'color': [255, 255, 0]}, 
    8: {'name': 'container/generic-object', 'color': [255, 0, 127]}, 
    9: {'name': 'asphalt', 'color': [64, 64, 64]}, 
    10: {'name': 'gravel', 'color': [255, 128, 0]}, 
    11: {'name': 'building', 'color': [255, 0, 0]}, 
    12: {'name': 'mulch', 'color': [153, 76, 0]}, 
    13: {'name': 'rock-bed', 'color': [102, 102, 0]}, 
    14: {'name': 'log', 'color': [102, 0, 0]}, 
    15: {'name': 'bicycle', 'color': [0, 255, 128]}, 
    16: {'name': 'person', 'color': [204, 153, 255]}, 
    17: {'name': 'fence', 'color': [102, 0, 204]}, 
    18: {'name': 'bush', 'color': [255, 153, 204]}, 
    19: {'name': 'sign', 'color': [0, 102, 102]}, 
    20: {'name': 'rock', 'color': [153, 204, 255]}, 
    21: {'name': 'bridge', 'color': [102, 255, 255]}, 
    22: {'name': 'concrete', 'color': [101, 101, 11]}, 
    23: {'name': 'picnic-table', 'color': [114, 85, 47]}
}

def main():
    output_dir = "./output/RUGD/masks/{}.png"
    color_masks_dir = pathlib.Path("./data/Rellis-3D/masks")
    color_masks = sorted(list(color_masks_dir.glob('*.png')))

    count = 1
    for color_mask in color_masks:
        mask = torch.tensor(np.array(Image.open(str(color_mask))))
        label = convert_mask(mask, False)

        reverted_mask = convert_mask(label, True)
        assert (reverted_mask.numpy() == mask.numpy()).all()
        save_label = Image.fromarray(label.numpy())
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
    main()
