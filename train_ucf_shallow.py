import os
import sys
from trainer_shallow import train_model
data_path = '../UCF_for_R21D'
im_root = '../UCF-101_of'
batch_size = 16
train_model(101, data_path, im_root,batch_size=batch_size, 
        path="model_data_shallow.pth.tar")
