from hd_utils import (Config, get_mlp, lin_interpolate, generate_mlp_from_weights, reconstruct_from_mlp)
import hydra
from omegaconf import DictConfig
from dataset import WeightDataset
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from transformer import Transformer
import os
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder
import torch
import pytorch_lightning as pl
import wandb
from os.path import join
from datetime import datetime
from torch.utils.data import DataLoader
from typing import List

@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs/experiment_configs/overfit_single_transformer",
    config_name="train_plane_ginr_single",
)

def main(cfg: DictConfig):
    #lin_interpolate_weight_space_experiment(cfg)
    #reconstruct_from_ginr_experiment(cfg)
    #reconstruct_from_hyperdiff_weights(cfg)
    overfit_with_transformer(cfg)

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
        is_ginr= True if Config.get("mlps_type") == "ginr" else False
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

def reconstruct_from_ginr_experiment(cfg: DictConfig):
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
        # COMMENTED FOR DEBUG PURPOSSES
        #train_object_names = set([str.split(".")[0] for str in train_object_names])
        # SET TRAIN OBJECT NAMES MANUALLY FOR DEBUG PURPOSSES FOR SINGLE OBJECT
        train_object_names = train_object_names.item().split(".")[0]
    
    # set train dataset
    train_dt = WeightDataset(
        mlps_folder_train,
        None, # wandb_logger,
        None, # model_dims
        mlp_kwargs,
        cfg,
        train_object_names,
        is_ginr= True if Config.get("mlps_type") == "ginr" else False
    )
    
    print(
        "Train dataset length: {}".format(
            len(train_dt)
        )
    )
    
    for w_i in train_dt:
        print(w_i[0].shape)
        break
    
    iterator_w = iter(train_dt)
    ginr_weights = next(iterator_w)[0]
    # generate mlp from weights
    mlp_from_ginr = generate_mlp_from_weights(ginr_weights, mlp_kwargs)
    
    input = {"coords": torch.asarray([-0.5, -0.5, -0.5])}
    #mlp_from_ginr(input)
    
    #############
    # Function to calculate the total parameter size of a model
    total_params = 0
    for param in mlp_from_ginr.parameters():
        total_params += param.numel()
    print(total_params, len(ginr_weights))
    
    # generate mesh from MLP
    sdf_decoder = SDFDecoder(
            mlp_from_ginr,
            None,
            "init_from_model",
            cfg,
        )
    
    vertices, faces, _ = sdf_meshing.create_mesh(
                            sdf_decoder,
                            os.path.join(
                                "./experiments",
                                f"ginr_reconstruction_ply"
                            ),
                            N=256,
                            level=0
                            if Config.config["mlp_config"]["params"]["output_type"] == "occ" and Config.config["mlp_config"]["params"]["out_act"] == "sigmoid"
                            else 0
                        )
    
    return vertices, faces

def reconstruct_from_hyperdiff_weights(cfg: DictConfig):
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
        # COMMENTED FOR DEBUG PURPOSSES
        #train_object_names = set([str.split(".")[0] for str in train_object_names])
        # SET TRAIN OBJECT NAMES MANUALLY FOR DEBUG PURPOSSES FOR SINGLE OBJECT
        train_object_names = train_object_names.item().split(".")[0]
    
    # set train dataset
    train_dt = WeightDataset(
        mlps_folder_train,
        None, # wandb_logger,
        None, # model_dims
        mlp_kwargs,
        cfg,
        train_object_names,
        is_ginr= True if Config.get("mlps_type") == "ginr" else False
    )
    
    print(
        "Train dataset length: {}".format(
            len(train_dt)
        )
    )
    
    for w_i in train_dt:
        print(w_i[0].shape)
        break
    
    iterator_w = iter(train_dt)
    weights = next(iterator_w)[0]
    # generate mlp from weights
    mlp = generate_mlp_from_weights(weights, mlp_kwargs)
    
    reconstruct_from_mlp(mlp=mlp, cfg=cfg, additional_experiment_specifiers=[cfg.get("mlps_type")])
    
def overfit_with_transformer(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    print(method)
    
    # WanDB init
    wandb.init(
        project="hyperdiffusion_overfit_transformer",
        dir=config["tensorboard_log_dir"],
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        tags=[Config.get("mode")],
        mode="disabled" if Config.get("disable_wandb") else "online",
        config=dict(config),
    )

    wandb_logger = WandbLogger()
    wandb_logger.log_text("config", ["config"], [[str(config)]])
    print("wandb", wandb.run.name, wandb.run.id)
    
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
        # COMMENTED FOR DEBUG PURPOSSES
        #train_object_names = set([str.split(".")[0] for str in train_object_names])
        # SET TRAIN OBJECT NAMES MANUALLY FOR DEBUG PURPOSSES FOR SINGLE OBJECT
        train_object_names = train_object_names.item().split(".")[0]
    
    # set train dataset
    train_dt = WeightDataset(
        mlps_folder_train,
        wandb_logger,
        None, # model_dims
        mlp_kwargs,
        cfg,
        train_object_names,
        is_ginr= True if Config.get("mlps_type") == "ginr" else False
    )
    
    print(
        "Train dataset length: {}".format(
            len(train_dt)
        )
    )

    # Initialize Transformer for HyperDiffusion
    if "hyper" in method:
        mlp = get_mlp(mlp_kwargs)
        state_dict = mlp.state_dict()
        layers = []
        layer_names = []
        for l in state_dict:
            shape = state_dict[l].shape
            layers.append(np.prod(shape))
            layer_names.append(l)
        transformer = Transformer(
            layers, layer_names, **Config.config["transformer_config"]["params"]
        ).cuda()
        
        # Specify where to save checkpoints
    checkpoint_path = join(
        config["tensorboard_log_dir"],
        "lightning_checkpoints",
        f"{str(datetime.now()).replace(':', '-') + '-' + wandb.run.name + '-' + wandb.run.id}",
    )
    
    transformer.set_mlp_kwargs(mlp_kwargs)
    transformer.set_cfg(cfg)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=Config.get("epochs"),
        strategy="ddp",
        logger=wandb_logger,
        default_root_dir=checkpoint_path,
        callbacks=[
            lr_monitor,
        ],
        check_val_every_n_epoch=Config.get("val_fid_calculation_period"),
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )
    
    train_dl = DataLoader(
            train_dt,
            batch_size=Config.get("batch_size"),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    
    if Config.get("mode") == "train":
        # If model_resume_path is provided (i.e., not None), the training will continue from that checkpoint
        trainer.fit(transformer, train_dl, train_dl)

if __name__ == "__main__":
    main()