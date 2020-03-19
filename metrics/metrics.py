import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def get_auc_score(y_true, y_pred):
  y_oh = np.zeros((y_true.shape[0], 4))
  for i, value in enumerate(y_true.cpu().numpy()):
    y_oh[i, value] = 1
  y_pred = y_pred.cpu().detach().numpy()
  try:
    roc_value = roc_auc_score(y_oh, y_pred)
  except:
    roc_value = 0
  return roc_value

if __name__ == "__main__":
    pass