import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.datasets import generate_train_validation_dataloader
from models.models import TL_EfficientNet
from models.controller import Controller
from utils.utils import get_device, read_parameters, separate_train_val
import albumentations as A
from albumentations import pytorch

if __name__ == "__main__":
    df = pd.read_csv("data/data/train.csv")
    device = get_device()
    config = read_parameters()
    train_df, val_df = separate_train_val(df, 0, 0.8)
    train_transform = A.Compose([
                        A.RandomResizedCrop(height = 512, width = 512, p = 1.0),
                        A.Flip(),
                        A.ShiftScaleRotate(rotate_limit = 1.0, p = 0.8),
                        A.OneOf([
                            A.IAAEmboss(p = 1.0),
                            A.IAASharpen(p = 1.0),
                            A.Blur(p = 1.0),
                            A.RGBShift(p = 1),
                            A.RandomBrightness(p = 1),
                            A.RandomContrast(p = 1)
                        ], p = 0.5),
                        A.OneOf([
                            A.ElasticTransform(p = 1.0),
                            A.IAAPiecewiseAffine(p = 1.0)
                        ], p = 0.5),
                        A.Normalize(p = 1.0),
                        pytorch.ToTensorV2(),
                            ])
    val_transform = A.Compose([
                        A.Resize(height = 512, width = 512, p = 1.0),
                        A.Normalize(p = 1.0),
                        pytorch.ToTensorV2(),
                          ])
    train_loader, val_loader = generate_train_validation_dataloader(train_df, 
                                                                    val_df, 
                                                                    config["train_parameters"]["batch_size"],
                                                                    "data/data/images/",
                                                                    train_transform,
                                                                    val_transform)
    EF_Net = TL_EfficientNet(config["network_parameters"], True).to(device)
    Optimizer = optim.Adam(EF_Net.parameters(), lr = config["train_parameters"]["learning_rate"])
    Loss = nn.CrossEntropyLoss()
    Train = Trainer(model = EF_Net,
                    optimizer = Optimizer,
                    loss = Loss,
                    train_data = train_loader,
                    val_data = val_loader,
                    epochs = config["train_parameters"]["epochs"],
                    device = device)
    Train.train()