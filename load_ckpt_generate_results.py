from hyperdiffusion import HyperDiffusion
from model import create_model
from omegaconf import DictConfig
import hydra
from hd_utils import Config, get_mlp
from transformer import Transformer
import numpy as np
import torch
import os
from dataset import ModulationDataset
from torch.utils.data import DataLoader

@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="train_plane_ginr_simplified",
)

def main(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    mlp_kwargs = None
    
    # Extract parameters from the command line or use default values
    level = cfg.get("level", 0)  # Default level
    checkpoint_path = cfg.get("checkpoint_path", "./save/epoch=3999-step=3999.ckpt")  # Default checkpoint path
    save_path = cfg.get("save_path", "./")  # Default path to save OBJ files

    # In HyperDiffusion, we need to know the specifications of MLPs that are used for overfitting
    if "hyper" in method:
        mlp_kwargs = Config.config["mlp_config"]["params"]
        
    # Initialize Transformer for HyperDiffusion
    if "hyper" in method:
        if "ginr_modulated" in method:
            mlp = create_model(config.arch)
            modulation_layer_shapes = []
            modulation_layer_names = []
            for modulated_param_id, modulated_param_name in enumerate(mlp.factors.modulated_param_names):
                modulated_param_shape = mlp.factors.init_modulation_factors[modulated_param_name].shape
                modulation_layer_shapes.append(np.prod(modulated_param_shape))
                modulation_layer_names.append(modulated_param_name)
            model = Transformer(
                modulation_layer_shapes, modulation_layer_names, **Config.config["transformer_config"]["params"]
            ).cuda()
        else:
            mlp = get_mlp(mlp_kwargs)
            state_dict = mlp.state_dict()
            layers = []
            layer_names = []
            for l in state_dict:
                shape = state_dict[l].shape
                layers.append(np.prod(shape))
                layer_names.append(l)
            model = Transformer(
                layers, layer_names, **Config.config["transformer_config"]["params"]
            ).cuda()
    
    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )
    train_object_names = set([str.split(".")[0] for str in train_object_names])
    
    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    mlps_folder_train = Config.get("mlps_folder_train")
    train_dt = ModulationDataset(
            mlps_folder_train,
            None,
            model.dims,
            mlp_kwargs,
            cfg,
            train_object_names,
            is_ginr= True if Config.get("mlps_type") == "ginr" else False
        )
    train_dl = DataLoader(
            train_dt,
            batch_size=Config.get("batch_size"),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    input_data = next(iter(train_dl))[0]
    # Initialize HyperDiffusion
    diffuser = HyperDiffusion(
        model=model,
        train_dt=train_dt,
        val_dt=None,
        test_dt=None,
        mlp_kwargs=mlp_kwargs,
        image_shape=input_data.shape,
        method=method,
        cfg=cfg
    )
    checkpoint = torch.load(checkpoint_path)
    diffuser.load_state_dict(checkpoint["state_dict"])
    
    # normalization
    if cfg.normalize_input:
        # calculate train set mean and std for normalization
        mean = 0.0
        no_samples = 0
        for i, batch_i in enumerate(train_dl):
            # batch_i[0].shape = [batch_size, weights_dim]
            batch_mean = batch_i[0].mean(dim=1).sum(dim=0)
            mean += batch_mean
            no_samples += batch_i[0].shape[0]
        mean /= no_samples
        print(no_samples, mean)
        
        std = 0.0
        for i, batch_i in enumerate(train_dl):
            batch_std = batch_i[0].std(dim=1).sum(dim=0)
            std += batch_std
        std /= no_samples
        diffuser.data_mean = mean
        diffuser.data_std = std
    
    level = cfg.get("level", 0)
    generate(diffuser=diffuser, level=level, save_path=save_path)
    
def generate(diffuser : HyperDiffusion, level = 0, save_path="./"):
    p_out = diffuser.diff.p_sample_loop(
                diffuser.model,
                shape=(5, *diffuser.image_size[1:])
            ).cpu().float()
    
    ddim_out = diffuser.diff.ddim_sample_loop(
                diffuser.model,
                shape=(5, *diffuser.image_size[1:])
            ).cpu().float()
    
    if diffuser.cfg.normalize_input:
        p_out = (p_out * diffuser.data_std.cpu()) + diffuser.data_mean.cpu()
        ddim_out = (ddim_out * diffuser.data_std.cpu()) + diffuser.data_mean.cpu()
        
    meshes_p_out, sdfs_p_out = diffuser.generate_meshes(p_out, None, res=256, level=level)
    meshes_ddim_out, sdfs_ddim_out = diffuser.generate_meshes(ddim_out, None, res=256, level=level)
    
    for index, mesh in enumerate(meshes_p_out):
        mesh = mesh.export(file_type='obj')
        with open(os.path.join(save_path, f'p_out_{index}_{level}.obj'), 'w') as file:
            file.write(mesh)
        
    for index, mesh in enumerate(meshes_ddim_out):
        mesh = mesh.export(file_type='obj')
        with open(os.path.join(save_path, f'ddim_{index}_{level}.obj'), 'w') as file:
            file.write(mesh)
    
if __name__ == "__main__":
    main()