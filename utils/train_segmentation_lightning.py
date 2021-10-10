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
from collections import Counter

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
import torchmetrics

from models.pointnet2_part_seg_msg import *

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
parser.add_argument('--stats', action='store_true', default=False)

RANDOM_SEED = 2  # fix seed
# print("Random Seed: ", RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

class LitPointNet(pl.LightningModule):

    def get_model(self, name):
        if name == "pointnet_dense_cls":
            classifier = PointNetDenseCls(k=self.num_classes, feature_transform=self.hparams.feature_transform)
            loss = torch.nn.NLLLoss(weight=self.hparams.class_weights)
        elif name == "pointnet2_part_seg":
            classifier = get_model(2, normal_channel=False)
            loss = torch.nn.NLLLoss(weight=self.hparams.class_weights)
            classifier = classifier.apply(weights_init)

        return classifier, loss

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

        self.hparams.class_weights = None
        self.model_name = "pointnet2_part_seg"
        #self.model_name = "pointnet_dense_cls"
        self.classifier, self.loss = self.get_model(self.model_name)

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        #self.hparams.class_weights = torch.Tensor([4.47,1], device=self.device)



        self.test_acc, self.val_acc, self.train_acc = torchmetrics.Accuracy(), torchmetrics.Accuracy(), torchmetrics.Accuracy()
        self.test_mIoU, self.val_mIoU, self.train_mIoU = torchmetrics.IoU(num_classes=self.num_classes), torchmetrics.IoU(num_classes=self.num_classes), torchmetrics.IoU(num_classes=self.num_classes)
        self.test_cIoU, self.val_cIoU, self.train_cIoU = torchmetrics.IoU(num_classes=self.num_classes, reduction="none"), torchmetrics.IoU(num_classes=self.num_classes, reduction="none"), torchmetrics.IoU(num_classes=self.num_classes, reduction="none")
        self.accuracy = {
            "train": self.train_acc, "val": self.val_acc, "test": self.test_acc
        }
        self.mIoU = {
            "train": self.train_mIoU, "val": self.val_mIoU, "test": self.test_mIoU
        }
        self.cIoU = {
            "train": self.train_cIoU, "val": self.val_cIoU, "test": self.test_cIoU
        }

        os.makedirs(self.hparams.outf, exist_ok=True)

        if args.stats:
            self.get_dataset_stats()

    def get_dataset_stats(self):
        targets = Counter({'1': 0, '2':0})
        for _, target in tqdm(self.test_dataloader()):
            unique, counts = torch.unique(target, return_counts=True)
            unique = [str(u.item()) for u in unique]
            targets = targets + Counter(dict(zip(unique, counts)))
            #targets[target] += 1
        print(targets)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # logger.debug(x.shape)
        if self.model_name == "pointnet2_part_seg":
            bs,_,_ = x.shape
            return self.classifier(x, torch.ones(bs,device=self.device))
        else:
            return self.classifier(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def predict(self, batch, set):
        points, target = batch
        points = points.transpose(2, 1)
        points, target = points, target

        pred, _ = self.forward(points)[:2]
        #print(pred.shape)
        pred = pred.contiguous().view(-1, self.num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        #print(pred.size(), target.size())
        loss = self.loss(pred, target)
        if self.hparams.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        self.log(f'{set}_loss', loss, on_epoch=True, on_step=True)

        self.accuracy[set](pred, target)
        self.mIoU[set](pred, target)
        cIoU = self.cIoU[set](pred, target)

        self.log(f'{set}_accuracy', self.accuracy[set], on_epoch=True, metric_attribute=f"{set}_acc")
        self.log(f'{set}_mIoU', self.mIoU[set], on_epoch=True, metric_attribute=f"{set}_mIoU")
        self.log(f'{set}_cIoU_1', cIoU[0], on_epoch=True)
        self.log(f'{set}_cIoU_2', cIoU[1], on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.predict(batch, set="val")

    def training_step(self, batch, batch_idx):
        return self.predict(batch, set="train")

    def test_step(self, batch, batch_idx):
        self.predict(batch, set="test")
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
            pred = self.forward(point)[0]
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
            batch_size=self.hparams.bs,
            shuffle=False,
            num_workers=self.hparams.workers)

if __name__ == '__main__':
    args = parser.parse_args()
    prefix = "pointnet"

    if args.viz:
        from show3d_balls import showpoints
        import matplotlib.pyplot as plt

    if args.test:
        pointnet_model = LitPointNet.load_from_checkpoint("lightning_logs/pointnet-last.ckpt")
        #pointnet_model.eval()
        trainer = pl.Trainer.from_argparse_args(args, accelerator="dp")
        trainer.test(pointnet_model)
    else:
        pointnet_model = LitPointNet(conf=args)
        checkpoint_callback = ModelCheckpoint(
            dirpath='lightning_logs',
            filename=prefix+'-{epoch}-{val_loss:.4f}',
            verbose=True,
            monitor='val_loss',
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
