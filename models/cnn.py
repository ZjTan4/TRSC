import torch
import torch.nn as nn

class CNN1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, image):
        pass

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
    
    def forward(self, x):
        pass