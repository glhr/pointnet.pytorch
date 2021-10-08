from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from show3d_balls import showpoints
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback

parser = argparse.ArgumentParser()
parser.add_argument(
    '--bs', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default="../shapenetcore_partanno_segmentation_benchmark_v0", help="dataset path")
parser.add_argument('--class_choice', type=str, default='inropa', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--npoints', type=int, default=2500, help='number of points per sample')
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--viz', action='store_true', default=False)
parser.add_argument('--show_gt', action='store_true', default=False)

RANDOM_SEED = 2  # fix seed
# print("Random Seed: ", RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class LitPointNet(pl.LightningModule):

    def __init__(self, conf, **kwargs):
        super().__init__()
        pl.seed_everything(RANDOM_SEED)
        self.save_hyperparameters(conf)

        self.train_dataset = ShapeNetDataset(
            root=self.hparams.dataset,
            classification=False,
            class_choice=[self.hparams.class_choice],
            data_augmentation=False,
            data='inropa' if self.hparams.class_choice == "inropa" else 'shuffled',
            npoints=self.hparams.npoints)
        self.test_dataset = ShapeNetDataset(
            root=self.hparams.dataset,
            classification=False,
            class_choice=[self.hparams.class_choice],
            split='test',
            data_augmentation=False,
            data='inropa' if self.hparams.class_choice == "inropa" else 'shuffled',
            npoints=self.hparams.npoints)

        self.num_classes = self.train_dataset.num_seg_classes

        self.classifier = PointNetDenseCls(k=self.num_classes, feature_transform=self.hparams.feature_transform)

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        os.makedirs(self.hparams.outf, exist_ok=True)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # logger.debug(x.shape)
        return self.classifier(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def predict(self, batch, set):
        points, target = batch
        points = points.transpose(2, 1)
        points, target = points, target

        pred, trans, trans_feat = self.forward(points)
        pred = pred.view(-1, self.num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if self.hparams.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).sum()
        accuracy = correct.item()/len(target)

        self.log(f'{set}_loss', loss, on_epoch=True, on_step=True)
        self.log(f'{set}_accuracy', accuracy, on_epoch=True)

        return loss

    def calc_iou(self,batch,set):
        points, target = batch
        points = points.transpose(2, 1)

        pred, _, _ = self.classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(self.num_classes)#np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            # print("part IoUs for class {}: {}".format(self.hparams.class_choice, part_ious))
            self.log(f'{set}_IoU_1', part_ious[0], on_epoch=True)
            self.log(f'{set}_IoU_2', part_ious[1], on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.predict(batch, set="val")

    def training_step(self, batch, batch_idx):
        return self.predict(batch, set="train")

    def test_step(self, batch, batch_idx):
        self.calc_iou(batch, set="test")
        if args.viz:
            point, seg = batch
            point = point[0]
            seg = seg[0]
            # print(point.size(), seg.size())
            point_np = point.cpu().numpy()

            cmap = plt.cm.get_cmap("hsv", 10)
            cmap = np.array([cmap(i) for i in range(10)])[:, :3]
            gt = cmap[seg.cpu().numpy() - 1, :]

            point = point.transpose(1, 0).contiguous()

            point = Variable(point.view(1, point.size()[0], point.size()[1]))
            pred, _, _ = self.classifier(point)
            pred_choice = pred.data.max(2)[1]
            # print(pred_choice)

            #print(pred_choice.size())
            pred_color = cmap[pred_choice.cpu().numpy()[0], :]

            #print(pred_color.shape)
            showpoints(point_np, c_gt=gt if args.show_gt else None, c_pred=pred_color)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.bs,
            shuffle=True,
            num_workers=self.hparams.workers,
            drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=self.hparams.workers,
            drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.workers)

if __name__ == '__main__':
    args = parser.parse_args()
    prefix = "pointnet"

    if args.test:
        pointnet_model = LitPointNet.load_from_checkpoint("lightning_logs/pointnet-epoch=530-val_loss=0.2393.ckpt")
        #pointnet_model.eval()
        trainer = pl.Trainer.from_argparse_args(args, accelerator="dp")
        trainer.test(pointnet_model)
    else:
        pointnet_model = LitPointNet(conf=args)
        checkpoint_callback = ModelCheckpoint(
            dirpath='lightning_logs',
            filename=prefix+'-{epoch}-{val_loss:.4f}',
            verbose=True,
            monitor='train_loss',
            mode='min',
            save_last=True
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = f"{prefix}-last"

        lr_monitor = LearningRateMonitor(logging_interval='step')

        callbacks = [lr_monitor,checkpoint_callback]

        wandb_logger = WandbLogger(project='scan-glowup', log_model = False, name = prefix)
        wandb_logger.log_hyperparams(pointnet_model.hparams)

        trainer = pl.Trainer.from_argparse_args(args,
                    check_val_every_n_epoch=1,
                    log_every_n_steps=1,
                    logger=wandb_logger,
                    checkpoint_callback=True,
                    callbacks=callbacks,
                    accelerator="dp")
        trainer.fit(pointnet_model)
