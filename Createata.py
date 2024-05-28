import os
import open3d as o3d
import numpy as np
from pathlib import Path

from utils import load_files_in_directory

def load_function(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    return mesh

if __name__ == "__main__":
    # Specify the directory path containing the files to load
    directory_path = 'Datasets/PUNet/meshes/train/'
    new_path = 'Datasets/PUNet/pointclouds/train/50000/'

    # Load every file in the specified directory
    loaded_files = load_files_in_directory(directory_path, load_function)

    # Process the loaded files as needed (e.g., print or use the loaded data)
    for filename, mesh in loaded_files.items():
        print(f"File '{filename}'")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(50000)

        out = np.append(pcd.points, pcd.normals, axis=1)

        np.savetxt(new_path + filename + ".xyz", out)