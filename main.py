from datetime import datetime
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from models.cnn import CNN1

from custom_dataset import CustomDatsetMemory, CustomDatsetIO

import matplotlib.pyplot as plt

# if not os.path.exists('./model'):
#     os.mkdir('./model')

use_cuda = True if torch.cuda.is_available() else False
if use_cuda:
    print('use cuda')
else:
    print('cpu')
device = torch.device("cuda:9" if use_cuda else "cpu")

# data_path = "./data/RUGD"
# img_ext = '.png'
data_path = "./data/Rellis-3D"
img_ext = '.jpg'

batch_size = 50
transform = transforms.Compose([
    transforms.ToTensor(), 
])

TRAIN_RATIO = 0.7
# dataset = CustomDatsetMemory(data_path)
dataset = CustomDatsetIO(data_path, img_ext=img_ext)
train_set, test_set = torch.utils.data.random_split(dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=True, shuffle=True)

print(len(train_loader))
print(len(test_loader))
num_epoch = 200

model = CNN1()
criterion = None
optimizer = torch.optim.SGD(
    params=dict(model.named_parameters()), lr=0.01, 
    # Optional Params
    momentum=0.9, 
    weight_decay=0.0001, 
    nesterov=False,
)

for epoch in range(num_epoch):
    start = datetime.now()
    for i, (img, mask) in enumerate(train_loader):
        # print(img.shape)
        # print(mask.shape)
        img = img.to(device)
        mask = mask.to(device)

        pred_mask = model(img)
        loss = criterion(pred_mask, mask)

        # plt.imshow(img[0])
        # plt.show()
        # plt.imshow(img[1])
        # plt.show()
        # plt.imshow(mask[0])
        # plt.show()
        # plt.imshow(mask[1])
        # plt.show()



        print(datetime.now() - start)
        start = datetime.now()
