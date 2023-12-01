import os
import sys
import shutil

# Check if enough arguments are provided
if len(sys.argv) != 8:
    print("Usage: python script.py <list_file_path> <obj_source_folder> <weights_source_folder> <pointcloud_source_folder> <obj_destination_folder> <weights_destination_folder> <pointcloud_destination_folder>")
    sys.exit(1)

# Assign paths from command-line arguments
list_file_path = sys.argv[1]
obj_source_folder_path = sys.argv[2]
weights_source_folder_path = sys.argv[3]
pointcloud_source_folder_path = sys.argv[4]
obj_destination_folder_path = sys.argv[5]
weights_destination_folder_path = sys.argv[6]
pointcloud_destination_folder_path = sys.argv[7]

# Ensure destination directories exist, create if they don't
os.makedirs(obj_destination_folder_path, exist_ok=True)
os.makedirs(weights_destination_folder_path, exist_ok=True)
os.makedirs(pointcloud_destination_folder_path, exist_ok=True)

# Function to find and copy matching files
def find_and_copy_files(source_folder, destination_folder, suffix, pattern_func):
    for file_name in os.listdir(source_folder):
        if file_name.endswith(suffix) and pattern_func(file_name):
            source_file_path = os.path.join(source_folder, file_name)
            destination_file_path = os.path.join(destination_folder, file_name)
            shutil.copy2(source_file_path, destination_file_path)

# Read the .lst file to get the base names for .obj files
with open(list_file_path, 'r') as file:
    obj_base_names = file.read().splitlines()

# Function to match .obj and point cloud files
def match_obj_files(name):
    return name in obj_base_names

# Function to match weight files
def match_weight_files(name):
    base_name_without_ext = [bn.split('.')[0] for bn in obj_base_names]
    return any(bn in name.split('_') for bn in base_name_without_ext)

# Function to match point cloud files
def match_pointcloud_files(name):
    base_name_without_ext = [bn.split('.')[0] for bn in obj_base_names]
    return any(bn in name.split('.') for bn in base_name_without_ext)

# Copy matching .obj files
find_and_copy_files(obj_source_folder_path, obj_destination_folder_path, '.obj', match_obj_files)

# Copy matching weight files
find_and_copy_files(weights_source_folder_path, weights_destination_folder_path, '.pth', match_weight_files)

# Copy matching point cloud files
find_and_copy_files(pointcloud_source_folder_path, pointcloud_destination_folder_path, '.npy', match_pointcloud_files)

# Copy the .lst file to the destination obj folder and destination pointcloud folder
shutil.copy2(list_file_path, obj_destination_folder_path)
shutil.copy2(list_file_path, pointcloud_destination_folder_path)

print("Selected .obj files, weight files, point cloud files, and the list file have been copied to the respective destination folders.")
