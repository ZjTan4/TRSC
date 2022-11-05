from datetime import datetime
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from custom_dataset import CustomDatsetMemory, CustomDatsetIO

import matplotlib.pyplot as plt

if not os.path.exists('./model'):
    os.mkdir('./model')

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    print('use cuda')
else:
    print('cpu')
device = torch.device("cuda:9" if use_cuda else "cpu")

data_path = "./data/bdd10k"
batch_size = 50
transform = transforms.Compose([
    transforms.ToTensor(), 
])

# dataset = CustomDatsetMemory(data_path)
dataset = CustomDatsetIO(data_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

num_epoch = 100
for epoch in range(num_epoch):
    start = datetime.now()
    for i, (img, mask) in enumerate(dataloader):
        print(img.shape)
        img = img.to(device)
        mask = mask.to(device)
        # tmp = img
        # plt.imshow(tmp[0])
        # plt.show()
        # plt.imshow(tmp[1])
        # plt.show()
        # plt.imshow(mask[0])
        # plt.show()
        # plt.imshow(mask[1])
        # plt.show()
        print(datetime.now() - start)
        start = datetime.now()
