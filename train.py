import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse

from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
import denoise_model
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l','--layers', nargs='+', help='<Required> Set flag', required=True)
    parser.parse_args()

    args = parser.parse_args()
    if args.layers:
        layer_sizes = [int(x) for x in args.layers]
        print("layer sizes", layer_sizes)
    else:
        print("error")
        exit()

    #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    #torch.set_float32_matmul_precision('high') # +25% performance

    max_time={"hours": 24}
    denoiser = denoise_model.ImMultiModuleFilter(patch_size = 1000, k = 12, layer_sizes = layer_sizes)
    #denoiser = denoise_model.ImMultiModuleFilter.load_from_checkpoint("Models/lightning_logs/version_112/checkpoints/epoch=3-step=1200.ckpt")



    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = L.Trainer(max_time=max_time, limit_train_batches=1200, limit_val_batches=100, default_root_dir="Models/", callbacks=[lr_monitor])
    trainer.fit(model=denoiser)

    trainer.save_checkpoint("Models/latest.ckpt")
