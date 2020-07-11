import argparse
import json
import os
import shutil
import time

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
import dataset
from model import SCNN
from model_ENET_SAD import ENet_SAD
from utils.tensorboard import TensorBoard
from utils.transforms import *
from utils.lr_scheduler import PolyLR


from multiprocessing import Process, JoinableQueue
from threading import Lock
import pickle
#from torch.multiprocessing import Process, SimpleQueue

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/test_finding_files")
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    return args
args = parse_args()

# ------------ config ------------
exp_dir = args.exp_dir
while exp_dir[-1]=='/':
    exp_dir = exp_dir[:-1]
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

device = torch.device(exp_cfg['device'])
tensorboard = TensorBoard(exp_dir)

# ------------ train data ------------
# # CULane mean, std
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
# mean=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)
transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(),
                          Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
train_dataset = Dataset_Type(Dataset_Path[dataset_name], "train", transform_train)
#train_loader = DataLoader(train_dataset, batch_size=exp_cfg['dataset']['batch_size'], shuffle=True, collate_fn=train_dataset.collate, num_workers=16)
train_file_check = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate, num_workers=16)

# ------------ val data ------------
transform_val_img = Resize(resize_shape)
transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_val = Compose(transform_val_img, transform_val_x)
val_dataset = Dataset_Type(Dataset_Path[dataset_name], "val", transform_val)
#val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=4)
val_file_check = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate, num_workers=4)

# ------------ preparation ------------
if exp_cfg['model'] == "scnn":
    net = SCNN(resize_shape, pretrained=True)
elif exp_cfg['model'] == "enet_sad":
    net = ENet_SAD(resize_shape, sad=True)
else:
    raise Exception("Model not match. 'model' in 'cfg.json' should be 'scnn' or 'enet_sad'.")

net = net.to(device)
net = torch.nn.DataParallel(net)

optimizer = optim.SGD(net.parameters(), **exp_cfg['optim'])
lr_scheduler = PolyLR(optimizer, 0.9, **exp_cfg['lr_scheduler'])
best_val_loss = 1e6


"""
def batch_processor(arg):
    b_queue, data_loader = arg
    while True:
        if b_queue.empty():
            sample = next(data_loader)
            b_queue.put(sample)
            b_queue.join()
"""

train_error = []
val_error = []

def finding_broken_file(train_file_check, val_file_check):
    print("train dataset checking")
    for batch_idx, sample in enumerate(train_file_check):
        print(sample['img_name'][0])
        if sample['img'][0] == None:
            train_error.append(sample['img_name'][0])
    print("validation dataset checking")
    for batch_idx, sample in enumerate(val_file_check):
        print(sample['img_name'][0])
        if sample['img'][0] == None:
            val_error.append(sample['img_name'][0])
    print("finish checking")

def main():
    global best_val_loss
    finding_broken_file(train_file_check, val_file_check)
    print("Error from train dataset: ",len(train_error))
    print("Error from validation dataset: ", len(val_error))
    print("The lists are: ",train_error, val_error)


if __name__ == "__main__":
    main()
