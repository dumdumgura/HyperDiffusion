import numpy as np
import trimesh
import torch
import os
from scipy.spatial import distance_matrix
from evaluation_metrics_3d import compute_all_metrics

from hd_utils import calculate_fid_3d

def load_point_cloud(file_path):
    # Load a 3D file and return its vertices as a point cloud
    mesh = trimesh.load_mesh(file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh.vertices

from scipy.spatial import KDTree

def chamfer_distance(set1, set2):
    # Compute Chamfer Distance using KD-trees for memory efficiency
    tree1 = KDTree(set1)
    tree2 = KDTree(set2)

    # Find the nearest neighbors and their distances
    distances1, _ = tree1.query(set2)
    distances2, _ = tree2.query(set1)

    # Compute the mean of the distances
    mean_dist1 = np.mean(distances1)
    mean_dist2 = np.mean(distances2)

    return mean_dist1 + mean_dist2

def normalize_point_cloud(point_cloud):
    # Translate centroid to the origin
    centroid = np.mean(point_cloud, axis=0)
    translated = point_cloud - centroid
    
    # Scale the point cloud to fit in a unit cube
    max_distance = np.max(np.sqrt(np.sum(translated**2, axis=1)))
    normalized = translated / max_distance
    return normalized

def load_and_normalize_point_cloud(file_path, n_points = None):
    # Load a 3D file, return its vertices as a point cloud, and normalize it
    mesh = trimesh.load_mesh(file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    point_cloud = mesh.vertices
    if n_points is not None:
        point_cloud = mesh.sample(n_points)
    normalized_point_cloud = normalize_point_cloud(point_cloud)
    return normalized_point_cloud

# Update the load_point_cloud call in the calculate_average_chamfer_distance function
def calculate_average_chamfer_distance(objects_folder, ground_truth_folder):
    objects_files = sorted([f for f in os.listdir(objects_folder) if f.endswith('.ply')])
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith('.obj')])

    total_chamfer_distance = 0
    for obj_file in objects_files:
        base_name = os.path.splitext(obj_file)[0]
        gt_file = base_name + '.obj'
        
        if gt_file in ground_truth_files:
            obj_path = os.path.join(objects_folder, obj_file)
            gt_path = os.path.join(ground_truth_folder, gt_file)

            obj_point_cloud = load_and_normalize_point_cloud(obj_path)
            gt_point_cloud = load_and_normalize_point_cloud(gt_path)

            cd = chamfer_distance(obj_point_cloud, gt_point_cloud)
            total_chamfer_distance += cd
        else:
            print(f"Warning: No ground truth match found for {obj_file}")

    average_cd = total_chamfer_distance / len(objects_files)
    return average_cd

def calculate_metrics(objects_folder, ground_truth_folder):
    objects_files = sorted([f for f in os.listdir(objects_folder) if f.endswith('.ply')])
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith('.obj')])

    obj_point_cloud_arr = []
    gt_point_cloud_arr = []
    for obj_file in objects_files:
        base_name = os.path.splitext(obj_file)[0]
        gt_file = base_name + '.obj'
        
        if gt_file in ground_truth_files:
            obj_path = os.path.join(objects_folder, obj_file)
            gt_path = os.path.join(ground_truth_folder, gt_file)

            obj_point_cloud = load_and_normalize_point_cloud(obj_path, n_points=10000)
            gt_point_cloud = load_and_normalize_point_cloud(gt_path, n_points=10000)
            
            obj_point_cloud_arr.append(torch.tensor(obj_point_cloud, dtype=torch.float32))
            gt_point_cloud_arr.append(torch.tensor(gt_point_cloud, dtype=torch.float32))
    
    obj_point_cloud_arr = torch.stack(obj_point_cloud_arr)
    gt_point_cloud_arr = torch.stack(gt_point_cloud_arr)
    fid = calculate_fid_3d(
            sample_pcs=obj_point_cloud_arr, ref_pcs=gt_point_cloud_arr
        )
    
    metrics = compute_all_metrics(
            sample_pcs=obj_point_cloud_arr, ref_pcs=gt_point_cloud_arr, batch_size=obj_point_cloud_arr.shape[0], 
        )
    
    return fid, metrics
    
# Example usage
objects_folder = './final/generated_objects_HYPERDIFFUSION'
ground_truth_folder = './final/gt_objects'
average_cd = calculate_average_chamfer_distance(objects_folder, ground_truth_folder)
fid, metrics = calculate_metrics(objects_folder, ground_truth_folder)
print(f'Average Chamfer Distance: {average_cd}')
print(f'FID: {fid}\tMetrics: {metrics}')
