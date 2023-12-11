from math import ceil

import numpy as np
import pyrender
import torch
import trimesh

from mlp_models import MLP, MLP3D, MLP3D_GINR
from Pointnet_Pointnet2_pytorch.log.classification.pointnet2_ssg_wo_normals import \
    pointnet2_cls_ssg
from siren.experiment_scripts.test_sdf import SDFDecoder
from siren import sdf_meshing
from torchmetrics_fid import FrechetInceptionDistance

import os
from omegaconf import DictConfig
from typing import List

# Using edited 2D-FID code of torch_metrics
fid = FrechetInceptionDistance(reset_real_features=True)


def calculate_fid_3d(
    sample_pcs,
    ref_pcs,
    wandb_logger,
    path="Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth",
):
    batch_size = 10
    point_net = pointnet2_cls_ssg.get_model(40, normal_channel=False)
    checkpoint = torch.load(path)
    point_net.load_state_dict(checkpoint["model_state_dict"])
    point_net.eval().to(sample_pcs.device)
    count = len(sample_pcs)
    for i in range(ceil(count / batch_size)):
        if i * batch_size >= count:
            break
        print(
            ref_pcs[i * batch_size : (i + 1) * batch_size].shape,
            i * batch_size,
            (i + 1) * batch_size,
        )
        real_features = point_net(
            ref_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fake_features = point_net(
            sample_pcs[i * batch_size : (i + 1) * batch_size].transpose(2, 1)
        )[2]
        fid.update(real_features, real=True, features=real_features)
        fid.update(fake_features, real=False, features=fake_features)

    x = fid.compute()
    fid.reset()
    print("x fid_value", x)
    return x


class Config:
    config = None

    @staticmethod
    def get(param):
        return Config.config[param] if param in Config.config else None


def state_dict_to_weights(state_dict):
    weights = []
    for weight in state_dict:
        weights.append(state_dict[weight].flatten())
    weights = torch.hstack(weights)
    return weights


def get_mlp(mlp_kwargs):
    if "model_type" in mlp_kwargs:
        if mlp_kwargs.model_type == "mlp_3d":
            mlp = MLP3D(**mlp_kwargs)
        elif mlp_kwargs.model_type == "mlp_3d_ginr":
            mlp = MLP3D_GINR(**mlp_kwargs)
    else:
        mlp = MLP(**mlp_kwargs)
    return mlp


def generate_mlp_from_weights(weights, mlp_kwargs):
    mlp = get_mlp(mlp_kwargs)
    state_dict = mlp.state_dict()
    weight_names = list(state_dict.keys())
    for layer_id, layer in enumerate(weight_names):
        val = state_dict[layer]
        num_params = np.product(list(val.shape))
        w = weights[:num_params]
        # if GINR
        if mlp_kwargs.model_type == "mlp_3d_ginr":
            w = w.view(val.shape[::-1]).T
        else:
            w = w.view(*val.shape)
        state_dict[layer] = w
        weights = weights[num_params:]
    assert len(weights) == 0, f"len(weights) = {len(weights)}"
    mlp.load_state_dict(state_dict)
    return mlp


def render_meshes(meshes):
    out_imgs = []
    for mesh in meshes:
        img, _ = render_mesh(mesh)
        out_imgs.append(img)
    return out_imgs


def render_mesh(obj):
    if isinstance(obj, trimesh.Trimesh):
        # Handle mesh rendering
        mesh = pyrender.Mesh.from_trimesh(
            obj,
            material=pyrender.MetallicRoughnessMaterial(
                alphaMode="BLEND",
                baseColorFactor=[1, 0.3, 0.3, 1.0],
                metallicFactor=0.2,
                roughnessFactor=0.8,
            ),
        )
    else:
        # Handle point cloud rendering, (converting it into a mesh instance)
        pts = obj
        sm = trimesh.creation.uv_sphere(radius=0.01)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
        tfs = np.tile(np.eye(4), (len(pts), 1, 1))
        tfs[:, :3, 3] = pts
        mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    eye = np.array([2, 1.4, -2])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    camera_pose = look_at(eye, target, up)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(800, 800)
    color, depth = r.render(scene)
    r.delete()
    return color, depth


# Calculate look-at matrix for rendering
def look_at(eye, target, up):
    forward = eye - target
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    camera_pose = np.eye(4)
    camera_pose[:-1, 0] = right
    camera_pose[:-1, 1] = up
    camera_pose[:-1, 2] = forward
    camera_pose[:-1, 3] = eye
    return camera_pose

def lin_interpolate(w_1, w_2, lerp_weight):
    lerp = torch.lerp(w_1, w_2, lerp_weight)
    return lerp

def reconstruct_from_mlp(mlp, cfg: DictConfig, additional_experiment_specifiers: List[str]):
    """
    Reconstructs a mesh from a given MLP (Multi-Layer Perceptron) model.

    This function takes an MLP model and configuration parameters to generate
    a mesh representation. It saves the generated mesh to a specified folder 
    with a filename based on the experiment's name and additional specifiers.

    Parameters:
    mlp: The Multi-Layer Perceptron model used for mesh reconstruction.
    cfg (DictConfig): A configuration object containing parameters like the 
                      save folder and experiment name.
    additional_experiment_specifiers (List[str]): A list of strings to be 
                                                  appended to the filename for 
                                                  additional specification.

    Returns:
    tuple: A tuple containing two elements:
           - vertices (ndarray): The vertices of the reconstructed mesh.
           - faces (ndarray): The faces of the reconstructed mesh.

    Raises:
    - Various exceptions related to file handling and mesh generation 
      depending on the internal implementation of the `sdf_meshing.create_mesh` 
      and `SDFDecoder` classes.

    Note:
    - The mesh is saved to the path constructed by joining `cfg.get("save_folder")` 
      and `cfg.get("experiment_name")` with the elements from 
      `additional_experiment_specifiers`.
    - The resolution and level of the mesh are determined based on the 
      configuration of the MLP model specified in `cfg`.

    Example:
    >>> vertices, faces = reconstruct_from_mlp(mlp_model, config, ["_spec1", "_spec2"])
    Mesh saved to 'path_to_save_folder/experiment_name_spec1_spec2'
    """
    # generate mesh from MLP
    sdf_decoder = SDFDecoder(
            mlp,
            None,
            "init_from_model",
            cfg,
        )
    
    folder_to_save = cfg.get("save_folder")
    filename_to_save = cfg.get("experiment_name")
    
    for element in additional_experiment_specifiers:
        filename_to_save += "_" + element
        
    vertices, faces, _ = sdf_meshing.create_mesh(
                            sdf_decoder,
                            os.path.join(
                                folder_to_save,
                                filename_to_save
                            ),
                            N=256,
                            level=0
                            if Config.config["mlp_config"]["params"]["output_type"] == "occ" and Config.config["mlp_config"]["params"]["out_act"] == "sigmoid"
                            else 0
                        )
    
    return vertices, faces