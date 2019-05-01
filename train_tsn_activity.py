import os
import sys
from network_tsn import TSNClassifier
from dataset_activity import VideoDatasetTSN as VideoDataset
from torch.utils.data import DataLoader
from trainer import train_model
train_list = '../activitynet1.3/training.txt'
val_list = '../activitynet1.3/validation.txt'
save_path = 'tsn_model_resnet101_from_scratch_activity.pth'
resize_width = 360
resize_height = 256
crop_size = 224
clip_len = 1
# build model
num_classes = 200
model = TSNClassifier(num_classes=num_classes, base_model='resnet101', pretrained=False)
# build dataset
train_dataloader = DataLoader(
        VideoDataset(train_list, resize_width=resize_width,
                resize_height=resize_height,
                crop_size=crop_size,
                clip_len = clip_len), 
        batch_size=16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(
        VideoDataset(val_list, resize_width=resize_width,
                resize_height=resize_height,
                crop_size=crop_size,
                clip_len=clip_len,
                mode='val'), 
        batch_size=2, num_workers=2)
# train model
train_model(model, train_dataloader, val_dataloader, num_epochs=80, path=save_path)
