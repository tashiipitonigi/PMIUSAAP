import numpy as np
import torch
from evaluate_2 import point_mesh_bidir_distance_single_unit_sphere
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import point_cloud_utils as pcu

if __name__ == "__main__":   
    filepath_mesh = "align/model.obj"
    filepath_cloud = "align/unaligned.xyz"

    verts_np, faces_np = pcu.load_mesh_vf(filepath_mesh)
    verts = torch.FloatTensor(verts_np).cuda()
    faces = torch.LongTensor(faces_np).cuda()
        
    pcl = np.loadtxt(filepath_cloud, dtype=np.float32)
    pcl = pcl - pcl.mean(axis=0)

    pcl_dec = pcl[np.random.choice(len(pcl), size=50000, replace=False)]

    best = {"score": 1e10, "scale": 1.0, "rotation": np.array([0, 0, 0]), "transform": verts_np.mean(axis=0)}

    for i in tqdm(range(10000)):
        scale = best["scale"] * np.random.normal(1, 0.05, size=(1))[0]
        rotation = best["rotation"] + np.random.normal(0, 0.05, size=(3)) # (ar parak mazu var iesprust lokalaja minimuma)
        transform = best["transform"] + np.random.normal(0, 0.05, size=(3))

        p_mod = R.from_euler("ZYX", rotation).apply(pcl_dec * scale) + transform
        p2f = point_mesh_bidir_distance_single_unit_sphere(pcl=torch.FloatTensor(p_mod).cuda(), verts=verts, faces=faces).item()  * 10000

        if p2f < best["score"]:
            best = {"score": p2f, "scale": scale, "rotation": rotation, "transform": transform}

    print("print score with 10000 random samples: ", best["score"])
    np.savetxt("align/aligned.xyz", R.from_euler("ZYX", best["rotation"]).apply(pcl * best["scale"]) + best["transform"])

