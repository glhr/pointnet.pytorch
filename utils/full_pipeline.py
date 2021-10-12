import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt

from utils.train_segmentation_lightning import *

class Scan(data.Dataset):
    def __init__(self,
                 file,
                 npoints=4000):
        self.npoints = npoints
        self.filelist = [file]

    def __getitem__(self, index):

        try:
            fn = self.filelist[index]
            point_set = np.loadtxt(fn).astype(np.float32)

            choice = np.random.choice(len(point_set), self.npoints, replace=True)
            #resample
            point_set = point_set[choice, :]

            point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            point_set = point_set / dist #scale

            point_set = torch.from_numpy(point_set)

            return point_set.unsqueeze(0)
        except Exception as e:
            print(f"Failed to load sample {self.filelist[index]}: {e}")

    def __len__(self):
        return len(self.filelist)

dataset = Scan(file="/home/gala/aidenmark/inropa-sandbox/scan-glowup/data/pairs/points/20211007_142026_012345.pts")
points = dataset[0]
print(points.shape)
points = points.transpose(2, 1)

pointnet_model = LitPointNet.load_from_checkpoint("lightning_logs/pointnet-epoch=31-val_loss=0.1596.ckpt", conf=args)
pred = pointnet_model(points)
