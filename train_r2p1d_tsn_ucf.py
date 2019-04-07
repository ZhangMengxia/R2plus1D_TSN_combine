import os
import sys
from network import R2Plus1DTSNClassifier
from dataset import VideoDatasetTSN as VideoDataset
from torch.utils.data import DataLoader
from trainer import train_model
data_path = '../UCF_for_R21D'
im_root = '../UCF-101_of'
save_path = 'r2p1d_tsn_model.pth'
# build model
num_classes = 101
model = R2Plus1DTSNClassifier(num_classes=num_classes)
# build dataset
train_dataloader = DataLoader(VideoDataset(data_path, im_root), batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(VideoDataset(data_path, im_root, mode='val'), batch_size=2, num_workers=2)
# train model
train_model(model, train_dataloader, val_dataloader, path=save_path)
