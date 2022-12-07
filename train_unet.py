from datetime import datetime
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn

from models.cnn import CNN1
from models.hrnet import HRNet
from models.unet import UNet
from custom_dataset import CustomDatsetMemory, CustomDatsetIO
from utils.validate import get_confusion_matrix, testval
import matplotlib.pyplot as plt

# if not os.path.exists('./model'):
#     os.mkdir('./model')

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    print('use cuda')
else:
    print('cpu')
device = torch.device("cuda:9" if use_cuda else "cpu")

data_path = "./data/RUGD"
img_ext = '.png'
# data_path = "./data/Rellis-3D"
# img_ext = '.jpg'

batch_size = 10
img_transform = transforms.Compose([
    # transforms.Resize((552, 688)),
    transforms.CenterCrop((544, 688)),
    transforms.ToTensor(), 
])
gt_transform = transforms.Compose([
    # transforms.Resize((552, 688)),
    transforms.CenterCrop((544, 688)),
    transforms.ToTensor(), 
])
TRAIN_RATIO = 0.7
# dataset = CustomDatsetMemory(data_path)
# dataset = CustomDatsetIO(data_path, img_ext=img_ext)
dataset = CustomDatsetIO(data_path, img_ext=img_ext, img_transform=img_transform, gt_transform=gt_transform)
train_set, test_set = torch.utils.data.random_split(dataset, [int(TRAIN_RATIO * len(dataset)), len(dataset) - int(TRAIN_RATIO * len(dataset))])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True, 
    pin_memory=True, prefetch_factor=4, num_workers=4,
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=True, shuffle=True, 
    pin_memory=True, prefetch_factor=4, num_workers=4,
)

print(len(train_loader))
print(len(test_loader))
num_epoch = 200

# model = CNN1().to(device)
# model = HRNet().to(device)
model = UNet(encoder_chs=(3, 16, 32, 64, 128), decoder_chs=(128, 64, 32, 16), num_class=6).to(device)
# model = UNet(encoder_chs=(3, 16, 32, 64), decoder_chs=(64, 32, 16), num_class=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(), lr=0.01, 
    # Optional Params
    momentum=0.9, 
    weight_decay=0.0001, 
    nesterov=False,
)
model.train()
for epoch in range(num_epoch):
    print("Epoch: {}".format(epoch + 1))
    start = datetime.now()
    total_loss = 0
    for i, (img, mask, name) in enumerate(train_loader):
        # print(img.shape)
        # print(mask.shape)
        # print(mask.unique())
        img = img.float().to(device)
        # mask = (mask.squeeze() * 256.4102564102564).long().to(device)
        mask = (mask.squeeze()).long().to(device)

        # img = img.permute(0, 3, 1, 2)
        pred_mask = model(img)
        # print(pred_mask.shape)
        loss = criterion(pred_mask, mask)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(mask.unique())
        # plt.imshow(img[0].permute(1, 2, 0).cpu())
        # plt.show()
        # plt.imshow(img[1].permute(1, 2, 0).cpu())
        # plt.show()
        # plt.imshow(mask[0].cpu())
        # plt.show()
        # plt.imshow(mask[1].cpu())
        # plt.show()
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(test_set, test_loader, model, device)
    print("Mean IoU: {}".format(mean_IoU))
    print("Mean Accuracy: {}".format(mean_acc))
    print("IoU: {}".format(IoU_array))
    print("Accuracy: {}".format(pixel_acc))
    print("Total Loss: {}".format(total_loss))
    print("Total Time: {}".format(datetime.now() - start))
        # start = datetime.now()
