import os
import sys
from network import R2Plus1DTSNClassifier
from dataset_activity import VideoDatasetTSN as VideoDataset
from torch.utils.data import DataLoader
from trainer import train_model
train_list = '../activitynet1.3/training.txt'
val_list = '../activitynet1.3/validation.txt'
save_path = 'r2p1d_tsn_model_activity.pth'
# build model
num_classes = 200
model = R2Plus1DTSNClassifier(num_classes=num_classes)
# build dataset
train_dataloader = DataLoader(
        VideoDataset(train_list), batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(
        VideoDataset(val_list, mode='val'), batch_size=2, num_workers=2)
# train model
train_model(model, train_dataloader, val_dataloader, num_epochs=70, path=save_path)
