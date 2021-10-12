import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data
from pointnet.model import PointNetDenseCls
import matplotlib.pyplot as plt

from utils.train_segmentation_lightning import *
from utils.process_predictions import *

from tqdm import tqdm
import gc

class Scan(data.Dataset):
    def __init__(self,
                 file,
                 npoints=4000):
        self.npoints = npoints
        self.file = file

        self.point_set = np.loadtxt(file).astype(np.float32)
        n_selections = int(len(self.point_set)/self.npoints)
        # print(f"{n_selections} sets from a total of {len(self.point_set)}")

        indices_orig = np.array(range(len(self.point_set)))
        indices = range(len(self.point_set))

        self.sets = []
        self.centers = list(range(n_selections))
        self.scales = list(range(n_selections))

        for n in range(n_selections):
            selected_idx = np.random.choice(range(len(indices)), self.npoints, replace=False)
            indices = np.delete(indices,selected_idx)
            self.sets.append(indices_orig[selected_idx])

    def __getitem__(self, index):

        choice = self.sets[index]
        #resample
        point_set = self.point_set[choice, :]

        center = np.expand_dims(np.mean(point_set, axis = 0), 0)
        point_set = point_set - center # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        self.centers[index] = torch.from_numpy(center).cuda()
        self.scales[index] = dist

        point_set = torch.from_numpy(point_set).cuda()

        return point_set.unsqueeze(0), self.centers[index], dist

    def __len__(self):
        return len(self.sets)

dataset = Scan(file="/home/gala/aidenmark/inropa-sandbox/scan-glowup/data/pairs/points/20211007_142026_012345.pts")
pointnet_model = LitPointNet.load_from_checkpoint("lightning_logs/pointnet-epoch=31-val_loss=0.1596.ckpt", conf=args).cuda()
pointnet_model.eval()
print(pointnet_model.device)

full_prediction = torch.zeros(size=(1, dataset.npoints*len(dataset), 2), device=pointnet_model.device)
full_cloud = torch.zeros(size=(1, dataset.npoints*len(dataset), 3), device=pointnet_model.device)

for n,sample in enumerate(tqdm(dataset)):
    # print(point_set.shape)
    point_set, center, scale = sample
    start_idx = n*dataset.npoints
    end_idx = start_idx + dataset.npoints

    point_set_orig = point_set * scale
    point_set_orig = point_set_orig + center


    full_cloud[:,start_idx:end_idx,:] = point_set_orig

    points = point_set.transpose(2, 1)
    with torch.no_grad():
        pred, _ = pointnet_model.forward(points)[:2]

    full_prediction[:,start_idx:end_idx,:] = pred


    #print(n, full_prediction[:,start_idx,:], full_cloud[:,start_idx,:])

    # full_prediction.append(pred)

point_np = full_cloud.squeeze(0).cpu().numpy()
#print(point_np.shape)
pred_choice = full_prediction.data.max(2)[1].cpu().numpy()
# print(pred_choice)

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
pred_color = cmap[pred_choice[0], :]

display_pointcloud(point_np, colors=gt if args.show_gt else pred_color, display=True)
extract_workpiece(point_np, pred_choice.swapaxes(0,1))

# npoints = 2
# point_set = range(10)
# n_selections = int(len(point_set)/npoints)
# print(f"{n_selections} sets from a total of {len(point_set)}")
#
# indices_orig = np.array(range(len(point_set)))
# indices = range(len(point_set))
# print(indices)
#
# sets = []
#
# for n in range(n_selections):
#
#     selected_idx = np.random.choice(range(len(indices)), npoints, replace=False)
#     print(f"--> {selected_idx}")
#     indices = np.delete(indices,selected_idx)
#     print(indices)
#     sets.append(indices_orig[selected_idx])
