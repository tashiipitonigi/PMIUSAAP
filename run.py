import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import scipy.spatial
import numpy as np
import scipy
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
import open3d as o3d
from tqdm import tqdm
from denoise_model import ImMultiModuleFilter, PointDataset
import argparse
from utils import lerp

def filter_pointcloud(checkpoint_path, pointcloud_points, iterations=1, filter_stride=1):
    #denoiser = ImMultiModuleFilter.load_from_checkpoint(checkpoint_path)
    denoiser = ImMultiModuleFilter.load_from_checkpoint(checkpoint_path)
    denoiser.eval()
    denoiser.freeze()
    denoiser.cuda()

    if denoiser.patch_size > pointcloud_points.shape[0]:
        raise Exception('point cloud doesnt have enough points for filtering. Has({}), Needs({})'.format(pointcloud_points.shape[0], denoiser.patch_size)) 

    plc_noisy_np = pointcloud_points.copy()

    for iter in range(iterations):
        kd_tree = scipy.spatial.KDTree(plc_noisy_np.copy())

        t = (1 / (iterations - iter)) / denoiser.patch_size * filter_stride
        print("t=", 1 / (iterations - iter))

        all_indexes = kd_tree.query(kd_tree.data, denoiser.patch_size)[1]

        for j in tqdm(range(0, plc_noisy_np.shape[0], filter_stride)):
            indexes = all_indexes[j]
            patch = kd_tree.data[indexes]
            patch, preprocess_data = PointDataset.Preprocess(patch)
            patch_tensor = torch.unsqueeze(torch.FloatTensor(patch), 0).cuda()

            offset = denoiser(patch_tensor)

            predicted_points = (patch_tensor + offset).cpu().numpy()
            predicted_points = PointDataset.InversePreprocess(predicted_points[0], preprocess_data)

            plc_noisy_np[indexes] = lerp(plc_noisy_np[indexes], predicted_points, t)

    return plc_noisy_np

def create_model_poisson(pointcloud_points, normal_k=15, depth=10, flip_normals=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_points)

    pcd.estimate_normals(fast_normal_computation=True)
    pcd.orient_normals_consistent_tangent_plane(k=normal_k)

    if flip_normals:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh.compute_vertex_normals()
    return mesh

def create_model_alpha(pointcloud_points, alpha=0.03):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_points)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--input_pointcloud', required=True, type=str)
    parser.add_argument('-c','--checkpoint', default="denoise_model.ckpt", type=str)
    parser.add_argument('-o','--output_dir', default="./", type=str)
    parser.add_argument('-d','--downsample_size', default=0.025, type=float)
    parser.add_argument('-i','--iterations', default=5, type=int)
    parser.add_argument('-i2','--true_iterations', default=1, type=int)
    parser.add_argument('-s','--stride', default=300, type=int)
    parser.add_argument('-n','--outliner_number', default=10, type=int)
    parser.add_argument('-r','--outliner_radius', default=0.07, type=float)

    parser.add_argument('-p','--poisson_depth', type=int) # default = 7
    parser.add_argument('-k','--normal_k', default=15, type=int)
    parser.add_argument('-f','--normal_flip', default=0, type=int)

    parser.add_argument('-a','--alpha_shape_alpha', type=float) # default = 0.2

    args = parser.parse_args()

    checkpoint = args.checkpoint
    pcd = None
    if(args.input_pointcloud[-3:] == "xyz"):
        p = np.loadtxt(args.input_pointcloud, dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
    else:
        pcd = o3d.io.read_point_cloud(args.input_pointcloud)

    downpcd = pcd.voxel_down_sample(voxel_size=args.downsample_size)
    #downpcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
    downpcd, _ = downpcd.remove_radius_outlier(nb_points=args.outliner_number, radius=args.outliner_radius)

    punkti_pirms = np.asarray(downpcd.points)
    punkti_pec = punkti_pirms
    for i in range(args.true_iterations):
        punkti_pec = filter_pointcloud(checkpoint, punkti_pec, args.iterations, args.stride)

    if args.poisson_depth is not None:
        #modelis_pirms = create_model_poisson(punkti_pirms, flip_normals=args.normal_flip, normal_k=args.normal_k,depth=args.poisson_depth)
        poisson_modelis_pec = create_model_poisson(punkti_pec, flip_normals=args.normal_flip, normal_k=args.normal_k,depth=args.poisson_depth)
        o3d.io.write_triangle_mesh(args.output_dir + "poisson.stl", poisson_modelis_pec)
    if args.alpha_shape_alpha is not None:
        #modelis_pirms = create_model_alpha(punkti_pirms, alpha=args.alpha_shape_alpha)
        alpha_modelis_pec = create_model_alpha(punkti_pec, alpha=args.alpha_shape_alpha)
        o3d.io.write_triangle_mesh(args.output_dir + "alpha.stl", alpha_modelis_pec)

    pcd_pec = o3d.geometry.PointCloud()
    pcd_pec.points = o3d.utility.Vector3dVector(punkti_pec)
    o3d.io.write_point_cloud(args.output_dir + "filtered.ply", pcd_pec)
    #np.savetxt(args.output_dir + "filtered_np.xyz", np.asarray(pcd_pec.points))

    o3d.visualization.draw_geometries([downpcd, pcd_pec.translate((6, 0, 0)), poisson_modelis_pec.translate((0, 6, 0)), alpha_modelis_pec.translate((6, 6, 0))])

