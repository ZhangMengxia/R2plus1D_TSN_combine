import os
import sys
from network_tsn import TSNClassifier
from dataset import VideoDatasetTSN as VideoDataset
from torch.utils.data import DataLoader
from trainer import train_model
data_path = '../UCF_for_R21D'
im_root = '../UCF-101_of'
save_path = 'tsn_model_resnet101_from_scratch.pth'
resize_width = 360
resize_height = 256
crop_size = 224
clip_len = 1
# build model
num_classes = 101
model = TSNClassifier(num_classes=num_classes, base_model='resnet101', pretrained=False)
# build dataset
train_dataloader = DataLoader(
        VideoDataset(data_path, im_root, resize_width=resize_width,
                resize_height=resize_height,
                crop_size=crop_size,
                clip_len = clip_len), 
        batch_size=16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(
        VideoDataset(data_path, im_root, resize_width=resize_width,
                resize_height=resize_height,
                crop_size=crop_size,
                clip_len=clip_len,
                mode='val'), 
        batch_size=2, num_workers=2)
# train model
train_model(model, train_dataloader, val_dataloader, path=save_path)
