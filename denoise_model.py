import os
import numpy as np
import lightning as L
from torch.utils.data import Dataset
import torch
import scipy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import preprocess
import model_arch

from torch import optim, nn, utils, Tensor

import random

# define the LightningModule
class ImMultiModuleFilter(L.LightningModule):
    def __init__(self, patch_size, k, layer_sizes):
        super().__init__()
        self.save_hyperparameters()
        self.patch_size = patch_size
        self.k = k
        self.layer_sizes = layer_sizes
        self.filter_module = model_arch.im_filter_module(input_dim=3, patch_nums=self.patch_size, k=self.k, layer_sizes=self.layer_sizes)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        clean_points = batch['pcl_clean']
        noisy_points = batch['pcl_noisy']
        noise_scale = batch['noise_scale']

        input_points = noisy_points
        target_points = clean_points

        output = self.filter_module(input_points)

        clean_nbs = target_points - input_points
        dist = ((output - clean_nbs)**2).sum(dim=-1)
        loss = dist.mean(dim=-1).mean(dim=-1)

        self.log("train_loss", loss)
        self.log("norm_train_loss", loss / noise_scale.sum())
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        clean_points = val_batch['pcl_clean']
        noisy_points = val_batch['pcl_noisy']
        noise_scale = val_batch['noise_scale']

        input_points = noisy_points
        target_points = clean_points

        output = self.filter_module(input_points)

        clean_nbs = target_points - input_points
        dist = ((output - clean_nbs)**2).sum(dim=-1)
        score = dist.mean(dim=-1).mean(dim=-1)

        self.log('val_loss', score)

        return score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( 
            self.filter_module.parameters(),
            lr=1e-4, 
            weight_decay=0
        )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, min_lr=1e-6),
            'interval': 'epoch',
            'frequency': 2,
            'monitor': 'train_loss',
        }
        return [optimizer], [scheduler]
    
    def forward(self, inputs):
        return self.filter_module(inputs)
    
    def train_dataloader(self):
        pointcloud_dataset = PointCloudDataset("Datasets", "PUNet", "train", 50000, None)
        point_dataset = PointDataset(pointcloud_dataset, "train", self.patch_size, None)
    
        return utils.data.DataLoader(point_dataset, shuffle=True, batch_size=4, num_workers=4, persistent_workers=True)
    
    def val_dataloader(self):
        pointcloud_dataset = PointCloudDataset("Datasets", "PUNet", "test", 50000, None)
        point_dataset = PointDataset(pointcloud_dataset, "test", self.patch_size, None)
    
        return utils.data.DataLoader(point_dataset, shuffle=False, batch_size=4, num_workers=1, persistent_workers=True)

class PointCloudDataset(Dataset):
    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        self.resolution = resolution
        self.split = split
        
        self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, str(resolution))
        for fn in os.listdir(self.pcl_dir):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            
            cloud = np.loadtxt(pcl_path, dtype=np.float32)

            # Specify column indices for each array
            indices_array1 = [0, 1, 2]  # Column indices for the first array
            indices_array2 = [3, 4, 5]  # Column indices for the second array

            # Extract the columns based on the specified indices
            pcl = cloud[:, indices_array1]  # Extract columns 0, 1, 2
            normals = cloud[:, indices_array2]  # Extract columns 3, 4, 5
            
            self.pointclouds.append({'pcl_clean': pcl, 'kd_tree': scipy.spatial.KDTree(pcl), 'normals': normals })
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __str__(self):
        return "Dataset with resolution: {}".format(self.resolution)

    def __getitem__(self, idx):
        data = {
                'pcl_clean': self.pointclouds[idx]['pcl_clean'], 
                'normals': self.pointclouds[idx]['normals'], 
                'kd_tree': self.pointclouds[idx]['kd_tree'], 
                'name': self.pointcloud_names[idx]
            }
        if self.transform is not None:
            data = self.transform(data)

        return data
    

def GetClosestPoints(kd_tree, point_index, point_count):
    nearest = kd_tree.data[kd_tree.query([kd_tree.data[point_index]], point_count, workers=4)[1]][0]
    return nearest

def GetClosestPointsWithNormals(normals, kd_tree, point_index, point_count):
    indexes = kd_tree.query([kd_tree.data[point_index]], point_count, workers=4)[1][0]
    nearest = kd_tree.data[indexes]
    norms = normals[indexes]
    return nearest, norms
   
class PointDataset(Dataset):
    def __init__(self, pointcloud_dataset: PointCloudDataset, split, patch_size=50, transform=None):
        super().__init__()
        self.pointcloud_dataset = pointcloud_dataset
        self.patch_size = patch_size
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.pointcloud_dataset) * self.pointcloud_dataset.resolution
    

    def Preprocess(point_patch):
        patch_diameter = float(np.linalg.norm(point_patch.max(0) - point_patch.min(0), 2))
        offset = np.mean(point_patch, axis=0)

        point_patch = (point_patch - offset) / patch_diameter
        point_patch, inverse_preprocess = preprocess.pca_alignment(point_patch)

        return point_patch, { "diameter" : patch_diameter, "offset" : offset, "inverse_matrix": inverse_preprocess}
    
    def PreprocessWithNoiseData(point_patch, preprocess_data):
        point_patch = (point_patch - preprocess_data["offset"]) / preprocess_data["diameter"]
        return np.array(np.linalg.inv(preprocess_data["inverse_matrix"]) * np.matrix(point_patch.T)).T
    
    def InversePreprocess(point_patch, preprocess_data):
        point_patch = np.array(preprocess_data["inverse_matrix"] * np.matrix(point_patch.T)).T
        return point_patch * preprocess_data["diameter"] + preprocess_data["offset"]
    
    def __getitem__(self, idx):
        cloud_index = int(idx / float(self.pointcloud_dataset.resolution))
        point_index = idx % self.pointcloud_dataset.resolution

        pcl_data = self.pointcloud_dataset[cloud_index]

        clean, normals = GetClosestPointsWithNormals(pcl_data['normals'], pcl_data['kd_tree'], point_index, self.patch_size)
        
        if self.split == 'train' or self.split == 'test':
            noise_scale = random.uniform(0.001, 0.015)
            noisy = clean + np.random.standard_normal(clean.shape) * noise_scale

            noisy, preprocess_data = PointDataset.Preprocess(noisy)
            clean = PointDataset.PreprocessWithNoiseData(clean, preprocess_data)

            data = {
                'pcl_noisy': torch.FloatTensor(noisy),
                'pcl_clean': torch.FloatTensor(clean),
                'normals': torch.FloatTensor(normals),
                'noise_scale': noise_scale,
            }
        else:
            print("unknown split")

        return data