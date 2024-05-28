import numpy as np
from pathlib import Path

import torch
import pytorch3d.loss
import point_cloud_utils as pcu

from tqdm import tqdm

from utils import load_files_in_directory

import argparse
import csv

def deconstruct_pcl_name(filepath):
    s = Path(filepath).stem.split("_")
    return {"name": s[0], "iterations": s[1], "stride": s[2]}

def load_pointclouds(filepath):
    pcl = np.loadtxt(filepath, dtype=np.float32)
    dc = deconstruct_pcl_name(filepath)
    dc["pcl"] = pcl
    return dc

def load_pointclouds_gt(filepath):
    pcl_clean_np = np.loadtxt(filepath, dtype=np.float32)   
    indices_array1 = [0, 1, 2]
    pcl_clean_np = pcl_clean_np[:, indices_array1] 
    return pcl_clean_np

def load_meshes_gt(filepath: str):
    verts, faces = pcu.load_mesh_vf(filepath)
    return {"verts": verts, "faces": faces}
     

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
    return (pc - center) / scale

# no iterativeofn
def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)

# no iterativeofn
def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls)

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--intermediate', required=True)
    args = parser.parse_args()

    intermediate_save_dir = args.intermediate
    
    pointclouds = load_files_in_directory(intermediate_save_dir, load_pointclouds)
    pointclouds_gt = load_files_in_directory("Datasets/PUNet/pointclouds/test/50000", load_pointclouds_gt)
    meshes_gt = load_files_in_directory("Datasets/PUNet/meshes/test", load_meshes_gt)
    #meshes_gt = load_files_in_directory("pedejais/ground_truth_mesh", load_meshes_gt)

    results = {}

    for filename, data in tqdm(pointclouds.items()):
        
        name = data["name"]
        iterations = data["iterations"]
        stride = data["stride"]

        pcl = data["pcl"]
        pcl_gt = pointclouds_gt[name]
        mesh_gt = meshes_gt[name]


        pcl_pred = torch.FloatTensor(pcl).unsqueeze(0).cuda()
        pcl_gt = torch.FloatTensor(pcl_gt).unsqueeze(0).cuda()

        verts = torch.FloatTensor(mesh_gt["verts"]).cuda()
        faces = torch.LongTensor(mesh_gt["faces"]).cuda()

        cd_sph = chamfer_distance_unit_sphere(pcl_pred, pcl_gt)[0].item()
        #cd_sph = 0

        p2f = point_mesh_bidir_distance_single_unit_sphere(pcl=pcl_pred[0], verts=verts, faces=faces).item()
        print(pcl_pred[0].shape)

        cd_sph *= 10000
        p2f *= 10000

        key = str(iterations) + " " + str(stride)
        if key not in results:
            results[key] = {"CD":0, "P2M":0, "Count":0}

        d = results[key]
        d["CD"] += cd_sph
        d["P2M"] += p2f
        d["Count"] += 1

    
    for k, v in results.items():
        print("settings It, St: {}, CD:{:.2f} P2M:{:.2f}".format(k, v["CD"] / v["Count"] * 10, v["P2M"] / v["Count"] * 10))

    with open(intermediate_save_dir + "results.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["Object", "CD", "P2M"])
        for k, v in results.items():
            print("settings It, St: {}, CD:{:.2f} P2M:{:.2f}".format(k, v["CD"] / v["Count"] * 10, v["P2M"] / v["Count"] * 10))
            spamwriter.writerow([k, v["CD"] / v["Count"] * 10, v["P2M"] / v["Count"] * 10])

        
