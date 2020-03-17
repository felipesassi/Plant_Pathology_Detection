import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

class Data_Generator(Dataset):
  def __init__(self, data, base_dir, transformer=None, train=True):
    self.data = data
    self.base_dir = base_dir  
    assert transformer != None, "Tranformer should be a valid value."
    self.transformer = transformer
    self.train = train
    if train == True:
        self.compute_labels()

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    img_name = self.data["image_id"][idx]
    directory = self.base_dir + img_name + ".jpg"
    img = plt.imread(directory)
    img = self.transformer(image = img)["image"]
    if self.train == True:
      img_class = self.labels[idx].reshape(1).astype(np.int64)
      return img, torch.from_numpy(img_class)
    else:
      return img

  def get_image_label(self, data):
    idx = 0
    for i, value in enumerate(data):
      if value == 1:
        idx = i
    return idx    

  def compute_labels(self): 
    self.labels = np.array(self.data[["healthy", "multiple_diseases", "rust", "scab"]].apply(self.get_image_label, axis = 1))
  
def generate_train_validation_dataloader(data_train, data_val, batch_size, base_dir, train_transform, val_transform):
    train_loader = DataLoader(Data_Generator(data_train, base_dir, train_transform, True), batch_size)
    validation_loader = DataLoader(Data_Generator(data_val, base_dir, val_transform, True), batch_size)
    return train_loader, validation_loader

if __name__ == "__main__":
    pass