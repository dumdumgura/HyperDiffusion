from hd_utils import (Config, get_mlp, lin_interpolate, generate_mlp_from_weights)
import hydra
from omegaconf import DictConfig
from dataset import WeightDataset
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from transformer import Transformer
import os
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder

@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="experiment",
)

def main(cfg: DictConfig):
    lin_interpolate_weight_space_experiment(cfg)

def lin_interpolate_weight_space_experiment(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    print(method)
    
    # Config set for weight reeading
    mlps_folder_train = Config.get("mlps_folder_train")
    
    # read mlp kwargs
    mlp_kwargs = Config.config["mlp_config"]["params"]
    
    #set train object names
    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )
    if not cfg.mlp_config.params.move:
        train_object_names = set([str.split(".")[0] for str in train_object_names])
    
    # set train dataset
    train_dt = WeightDataset(
        mlps_folder_train,
        None, # wandb_logger,
        None, # model_dims
        mlp_kwargs,
        cfg,
        train_object_names,
    )
    
    print(
        "Train dataset length: {}".format(
            len(train_dt)
        )
    )
    
    for w_i in train_dt:
        print(w_i[0].shape)
        break
    
    # lin. interpolate weight
    iterator_w = iter(train_dt)
    w_start = next(iterator_w)[0]
    w_end = next(iterator_w)[0]
    lin_erp_interval = 0.5
    w_lerp = lin_interpolate(w_start, w_end, 0.5)
    
    # generate mlp from weights
    mlp_lerp = generate_mlp_from_weights(w_lerp, mlp_kwargs)
    print(mlp_lerp)
    
    # generate mesh from MLP
    sdf_decoder = SDFDecoder(
            mlp_lerp,
            None,
            "init_from_model",
            cfg,
        )
    
    vertices, faces, _ = sdf_meshing.create_mesh(
                            sdf_decoder,
                            os.path.join(
                                "./experiments",
                                f"lin_erp_{lin_erp_interval}_ply"
                            ),
                            N=256,
                            level=0
                            if Config.config["mlp_config"]["params"]["output_type"] == "occ" and Config.config["mlp_config"]["params"]["out_act"] == "sigmoid"
                            else 0
                        )
    
    return vertices, faces

if __name__ == "__main__":
    main()