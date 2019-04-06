import os
import sys
from trainer import train_model
data_path = '../UCF_for_R21D'
im_root = '../UCF-101_of'
train_model(101, data_path, im_root)
