import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.datasets import generate_train_validation_dataloader, generate_test_dataloader
from models.models import TL_EfficientNet
from models.trainer import Controller
from utils.utils import get_device, read_parameters, separate_train_val
import albumentations as A
from albumentations import pytorch

if __name__ == "__main__":
    df = pd.read_csv("data/data/test.csv")
    device = get_device()
    config = read_parameters()
    test_transform = A.Compose([
                        A.Resize(height = 512, width = 512, p = 1.0),
                        A.Normalize(p = 1.0),
                        pytorch.ToTensorV2(),
                          ])
    test_loader = generate_test_dataloader(df, 
                                            config["train_parameters"]["batch_size"],  
                                            "data/data/images/", 
                                            test_transform, )
    EF_Net = TL_EfficientNet(config["network_parameters"], True).to(device)
    Optimizer = optim.Adam(EF_Net.parameters(), lr = config["train_parameters"]["learning_rate"])
    Loss = nn.CrossEntropyLoss()
    Control = Controller(model = EF_Net, device = device)
    Control.load()
    y = Control.evaluate()
    df["healthy"] = y.cpu().numpy()[:, 0]
    df["multiple_diseases"] = y.cpu().numpy()[:, 1]
    df["rust"] = y.cpu().numpy()[:, 2]
    df["scab"] = y.cpu().numpy()[:, 3]
    df.to_csv("sub.csv", index = False)