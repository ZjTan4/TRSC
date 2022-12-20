from datetime import datetime
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn

from models.cnn import CNN1
from models.hrnet import HRNet
from models.unet import UNet, UNet_GLCM
from custom_dataset import CustomDatsetMemory, CustomDatsetIO
from utils.validate import *
from utils.visualize import visualize
import matplotlib.pyplot as plt

# if not os.path.exists('./model'):
#     os.mkdir('./model')

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    print('use cuda')
else:
    print('cpu')
device = torch.device("cuda:5" if use_cuda else "cpu")
# device = torch.device("cuda:9" if use_cuda else "cpu")


data_path = "./data/RUGD"
img_ext = '.png'
# data_path = "./data/Rellis-3D"
# img_ext = '.jpg'

batch_size = 10
img_size = (552, 688)
# img_size = (128, 128)
img_transform = transforms.Compose([
    transforms.Resize(img_size),
    # transforms.Grayscale(num_output_channels=1), 
    # transforms.CenterCrop((544, 688)),
    transforms.ToTensor(), 
])
gt_transform = transforms.Compose([
    transforms.Resize(img_size),
    # transforms.CenterCrop((544, 688)),
    transforms.ToTensor(), 
])
TRAIN_RATIO = 0.7
# dataset = CustomDatsetMemory(data_path)
# dataset = CustomDatsetIO(data_path, img_ext=img_ext)
dataset = CustomDatsetIO(data_path, img_ext=img_ext, img_transform=img_transform, gt_transform=gt_transform)
trset, teset, _ = torch.utils.data.random_split(dataset, [int(TRAIN_RATIO * len(dataset)), int(len(dataset)) - int(TRAIN_RATIO * len(dataset)), len(dataset) - int(len(dataset))])
trloader = torch.utils.data.DataLoader(trset, batch_size=batch_size, drop_last=True, shuffle=True, 
    pin_memory=True, prefetch_factor=4, num_workers=4,
)
teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, drop_last=True, shuffle=True, 
    pin_memory=True, prefetch_factor=4, num_workers=4,
)

print("train loader batches: {}".format(len(trloader)))
print("test loader batches: {}".format(len(teloader)))
num_epoch = 200

# model = CNN1().to(device)
# model = HRNet().to(device)
# model = UNet(encoder_chs=(3, 16, 32, 64, 128), decoder_chs=(128, 64, 32, 16), num_class=6).to(device)
# model = UNet(encoder_chs=(3, 16, 32, 64), decoder_chs=(64, 32, 16), num_class=6).to(device)
model = UNet_GLCM(encoder_chs=(12, 16, 32, 64, 128), decoder_chs=(128, 64, 32, 16), num_class=6, device=device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(), lr=0.0001, 
    # Optional Params
    momentum=0.9, 
    weight_decay=0.0001, 
    nesterov=False,
)
model.train()
IoUs = []
for epoch in range(num_epoch):
    print("Epoch: {}".format(epoch + 1))
    start = datetime.now()
    total_loss = 0
    for i, (img, mask, name) in enumerate(trloader):
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

        # if epoch == 2 and i == 200:
        #     vis = torch.argmax(pred_mask[0], dim=0)
        #     visualize(vis.cpu())

        # print(mask.unique())
        # plt.imshow(img[0].permute(1, 2, 0).cpu())
        # plt.show()
        # plt.imshow(img[1].permute(1, 2, 0).cpu())
        # plt.show()
        # plt.imshow(mask[0].cpu())
        # plt.show()
        # plt.imshow(mask[1].cpu())
        # plt.show()
    confusion_matrix = testval(trset, trloader, model, device)
    mean_acc, pixel_acc = get_Acc(confusion_matrix)
    mean_IoU, IoU_array = get_IoU(confusion_matrix)
    IoUs.append(mean_IoU)
    print(confusion_matrix)
    print("Mean IoU: {}".format(mean_IoU))
    print("Mean Accuracy: {}".format(mean_acc))
    print("IoU: {}".format(IoU_array))
    print("Accuracy: {}".format(pixel_acc))
    print("Total Loss: {}".format(total_loss))
    print("Total Time: {}".format(datetime.now() - start))
    # if epoch == 199:
    #     image, label, name = trloader[0]
    #     pred = model(image)
    #     pred = torch.argmax(pred, dim=1)
    #     for i in range(4):
    #         plt.subplot(2, 4, i + 1)
    #         plt.show(visualize(pred[i]))
    #         plt.subplot(2, 4, i + 5)
    #         plt.show(label(pred[i]))
        # start = datetime.now()
# print(IoUs)
print("GLCM")
torch.save(model.state_dict(), "model_GLCM.pt")

# torch.save(model.state_dict(), "model.pt")

# plt.show()