import os
import sys
from network import R2Plus1DTSNClassifier
from dataset import VideoDatasetTSN as VideoDataset
from torch.utils.data import DataLoader
from trainer import test_model
data_path = '../UCF_for_R21D'
im_root = '../UCF-101_of'
save_path = 'r2p1d_tsn_model_ucf.pth'
# build model
num_classes = 101
model = R2Plus1DTSNClassifier(num_classes=num_classes)
# build dataset
val_dataloader = DataLoader(VideoDataset(data_path, im_root, mode='val'), batch_size=1, num_workers=2)
# test model
test_model(model, val_dataloader, path=save_path)
