from pathlib import Path
import torch
from PIL import Image
import numpy as np
import os

class CustomDatsetMemory(torch.utils.data.Dataset):
    def __init__(self, root=None, img_transform=None, gt_transform=None, img_ext='.jpg', gt_ext='.png'):
        self.root = root
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        image_dir = Path('{}/images'.format(self.root))
        images = sorted(list(image_dir.glob('*{}'.format(img_ext))))
        mask_dir = Path('{}/masks'.format(self.root))
        masks = sorted(list(mask_dir.glob('*{}'.format(gt_ext))))

        assert self.root is not None
        assert len(images) == len(masks) and len(images) > 0
        assert all(images[i].stem == masks[i].stem for i in range(len(images))) 

        self.images = read_images(images)
        self.masks = read_images(masks)
        # print(self.images.shape)
        # print(self.masks.shape)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.gt_transform is not None:
            mask = self.gt_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.images)

class CustomDatsetIO(torch.utils.data.Dataset):
    def __init__(self, root=None, img_transform=None, gt_transform=None, img_ext='.jpg', gt_ext='.png'):
        self.root = root
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        image_dir = Path('{}/images'.format(self.root))
        images = sorted(list(image_dir.glob('*{}'.format(img_ext))))
        mask_dir = Path('{}/masks'.format(self.root))
        masks = sorted(list(mask_dir.glob('*{}'.format(gt_ext))))

        assert self.root is not None
        assert len(images) == len(masks) and len(images) > 0
        assert all(images[i].stem == masks[i].stem for i in range(len(images))) 

        self.images = images
        self.masks = masks

    def __getitem__(self, index):
        image = Image.open(str(self.images[index]))

        mask = Image.open(str(self.masks[index]))

        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.gt_transform is not None:
            mask = self.gt_transform(mask)

        return image, (mask * 256).long(), str(self.masks[index].name)

    def __len__(self):
        return len(self.images)
    
    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

def read_images(image_paths):
    frame = len(image_paths)
    im = torch.tensor(np.array(Image.open(str(image_paths[0]))))
    byte = 1
    if len(im.shape) == 3:
        row, column, byte = im.shape
    else:
        row, column = im.shape

    images = torch.empty([frame, row, column, byte], dtype=im.dtype).squeeze()
    for i in range(frame):
        print("loading {} file: {}".format(i + 1, image_paths[i].stem), end = '\r')
        im = torch.tensor(np.array(Image.open(str(image_paths[i]))))
        images[i] = im
    print("")
    return images
