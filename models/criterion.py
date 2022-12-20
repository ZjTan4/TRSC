import torch
import torch.nn as nn

class CrossEntropy(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, prediction, target):
        # TODO
        pass