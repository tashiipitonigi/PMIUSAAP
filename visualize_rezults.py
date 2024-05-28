import os
import scipy.spatial
import torch
import numpy as np
import open3d as o3d

import chamferdist
import scipy

# nevar lietot torch geomettic chamferdist tadel jacer ka sis dod tadu pasu rezultatu

def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale

def normalize_pcl(pc, center, scale):
    return pc * scale + center

def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean', bidirectional = True):
    chamferDist = chamferdist.ChamferDistance()
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    return chamferDist(gen, ref, bidirectional)

def CreateModel(pcd):
    pcd.estimate_normals(fast_normal_computation=True)
    pcd.orient_normals_consistent_tangent_plane(k=15)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    mesh.compute_vertex_normals()
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.3)

    return mesh

def MapErrorToColor(error):
    return np.array([[0,0, 1 - max(0.0, min(1.0, error * 10)) ** 0.5]])

def CreatePointcloud(points, ground_truth):
    tree = scipy.spatial.KDTree(ground_truth)

    colors = np.zeros((0, 3))
    closest = tree.query(points, 1)
    for i in range(len(points)):
        d = np.linalg.norm(tree.data[closest[1][i]] - points[i] ,ord=2)
        colors = np.append(colors, MapErrorToColor(d), axis=0)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


result_dir = 'Results/Evaluation/20h/viz/'

p1 = np.loadtxt(result_dir + "clean.xyz", dtype=np.float32)
p1 = p1[:,:3]
p2 = np.loadtxt(result_dir + "noise_after.xyz", dtype=np.float32) 

c1 = CreatePointcloud(p1 + np.array([0,0,0]), p1 + np.array([0,0,0]))
c2 = CreatePointcloud(p2 + np.array([2,0,0]), p1 + np.array([2,0,0]))

o3d.visualization.draw_geometries([c1,c2])

