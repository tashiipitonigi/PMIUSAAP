import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse

import numpy as np
import scipy
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import open3d as o3d
from tqdm import tqdm

from run import filter_pointcloud

def create_pcl_name(name, iterations, stride):
    return "{}_{}_{}".format(name, iterations, stride)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--checkpoint', required=True)
    parser.add_argument('-i','--intermediate', required=True)
    parser.add_argument('-n','--noise_scale', default=0.015, type=float)
    args = parser.parse_args()

    #checkpoint = "Results/1000x2/version_71/checkpoints/epoch=74-step=45000.ckpt"
    checkpoint = args.checkpoint;
    result_save_dir = args.intermediate
    noise_scale = args.noise_scale

    os.makedirs(os.path.dirname(result_save_dir), exist_ok=True)

    objects = ["camel", "chair", "cow", "genus3", "Icosahedron", "elephant", "sculpt"]
    #settings = {5:[100, 300, 700, 100], 1:[100, 300, 700, 100]}
    
    settings = {1:[10, 100, 300, 700, 1000],
                2:[10, 100, 300, 700, 1000],
                3:[    100, 300, 700, 1000],
                4:[    100, 300, 700, 1000],
                5:[    100, 300, 700, 1000],
                }

    expected_time = 0
    for iterations, strides in settings.items():
        for stride in strides:
            expected_time += 50000.0/stride / 8.0 * iterations * len(objects)

    print(str(expected_time / 3600.0) + "h")
    print("noise ", noise_scale)

    for object in objects:
        print(object)
        pcl_clean_np = np.loadtxt("Datasets/PUNet/pointclouds/test/50000/" + object + ".xyz", dtype=np.float32)   
        indices_array1 = [0, 1, 2]
        pcl_clean_np = pcl_clean_np[:, indices_array1] 

        #scale = (torch.FloatTensor(pcl_clean_np) ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]
        #scale = scale.cpu().numpy()[0][0]
        scale = 1 # visi jau ir normalizeti

        noise = noise_scale # pie variance=1 standart deviation=1, tadel nekas nav jamaina
        # tika parbauits ka dod vienadus rezultatus ar IterativePFN pie tiem pasiem noise limeniem
        plc_noisy_np = pcl_clean_np + np.random.standard_normal(pcl_clean_np.shape) * noise * scale

        for iterations, strides in settings.items():
            for stride in strides:
                filered = filter_pointcloud(checkpoint, plc_noisy_np, iterations, stride)
                #for i in range(iterations):
                #    filered = filter_pointcloud(checkpoint, filered, 1, stride)
                np.savetxt(result_save_dir + create_pcl_name(object, iterations, stride), filered)