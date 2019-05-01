import os
import sys
from network_tsn import TSNClassifier
from dataset import VideoDatasetTSN as VideoDataset
from torch.utils.data import DataLoader
from trainer import test_model
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
val_dataloader = DataLoader(
        VideoDataset(data_path, im_root, resize_width=resize_width,
                resize_height=resize_height,
                crop_size=crop_size,
                clip_len=clip_len,
                mode='val'), 
        batch_size=1, num_workers=2)
# train model
test_model(model, val_dataloader, path=save_path)
