import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

if not os.path.exists('./model'):
    os.mkdir('./model')

cuda = True if torch.cuda.is_available() else False
if cuda:
    print('cuda')
else:
    print('cpu')

data_path = "./data"
batch_size = 200
transform = transforms.Compose([
    transforms.ToTensor(), 
])
dataset = dset.ImageFolder(data_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)


num_epoch = 100
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        pass
