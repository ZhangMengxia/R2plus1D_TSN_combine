import os
import sys
from network import R2Plus1DTSNClassifier
from dataset_activity import VideoDatasetTSN as VideoDataset
from torch.utils.data import DataLoader
from trainer import test_model
val_list = '../activitynet1.3/validation.txt'
save_path = 'r2p1d_tsn_model_activity.pth'
# build model
num_classes = 200
model = R2Plus1DTSNClassifier(num_classes=num_classes)
# build dataset
val_dataloader = DataLoader(VideoDataset(val_list, mode='val'), 
batch_size=1, num_workers=2)
# test model
test_model(model, val_dataloader, path=save_path)
