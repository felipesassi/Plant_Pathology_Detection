import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class TL_EfficientNet(nn.Module):
    def __init__(self, params_dict, pretrained=False):
        super().__init__()
        self.EFNet = EfficientNet.from_pretrained("efficientnet-b4")
        if pretrained == True:
            for parameter in self.EFNet.parameters():
                parameter.requires_grad = False
        self.in_features = self.EFNet._fc.in_features
        self.Linear = nn.Linear(self.in_features, params_dict["linear_1"])
        self.EFNet._fc = self.Linear
    
    def forward(self, x):
        x = self.EFNet(x)
        return x

if __name__ == "__main__":
    pass