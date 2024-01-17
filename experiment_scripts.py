from hd_utils import (Config, get_mlp, lin_interpolate, generate_mlp_from_weights, reconstruct_from_mlp, reconstruct_shape)
import hydra
from omegaconf import DictConfig
from dataset import WeightDataset, ModulationDataset
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from model import create_model
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
import math

@hydra.main(
    version_base=None,
    #config_path="configs/diffusion_configs/",
    #config_name="train_plane_ginr_simplified_relu",
    config_path="configs/overfitting_configs/",
    config_name="overfit_plane_ginr_simplified_relu",
    #config_name="overfit_plane_ginr_simplified",
)

def main(cfg: DictConfig):
    #lin_interpolate_weight_space_experiment(cfg)
    #reconstruct_from_ginr_experiment(cfg)
    #reconstruct_from_hyperdiff_weights(cfg)
    #overfit_with_transformer(cfg)
    #overfit_modulation_with_transformer(cfg)
    #lin_interpolate_weight_space_experiment_modulation(cfg)
    effect_of_noise_experiment(cfg)

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
    lin_erp_interval = 0.4
    w_lerp = lin_interpolate(w_start, w_end, lin_erp_interval)
    
    # generate mlp from weights
    mlp_lerp = generate_mlp_from_weights(w_lerp, mlp_kwargs, config=Config.config)
    mlp_lerp.to(torch.device("cuda"))
    print(mlp_lerp)
    
    meshes = mlp_lerp.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="lin_interpolate_weight_space_experiment" + str(lin_erp_interval))

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
        train_object_names = set([str.split(".")[0] for str in train_object_names])
        # SET TRAIN OBJECT NAMES MANUALLY FOR DEBUG PURPOSSES FOR SINGLE OBJECT
        #train_object_names = train_object_names.item().split(".")[0]
    
    # set train dataset
    train_dt = WeightDataset(
        mlps_folder_train,
        None, # wandb_logger,
        None, # model_dims
        mlp_kwargs,
        cfg,
        train_object_names,
        is_ginr = True if Config.get("mlps_type") == "ginr" else False
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
    mlp_from_ginr = generate_mlp_from_weights(ginr_weights, mlp_kwargs, config=Config.config)
    mlp_from_ginr.to(torch.device("cuda"))
    
    meshes = mlp_from_ginr.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="reconstruct_from_ginr_experiment")
    

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
        
def overfit_modulation_with_transformer(cfg: DictConfig):
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
    train_dt = ModulationDataset(
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

    # Initialize Transformer for HyperDiffusion
    mlp = create_model(config.arch)
    modulation_layer_shapes = []
    modulation_layer_names = []
    for modulated_param_id, modulated_param_name in enumerate(mlp.factors.modulated_param_names):
        modulated_param_shape = mlp.factors.init_modulation_factors[modulated_param_name].shape
        modulation_layer_shapes.append(np.prod(modulated_param_shape))
        modulation_layer_names.append(modulated_param_name)
    transformer = Transformer(
        modulation_layer_shapes, modulation_layer_names, **Config.config["transformer_config"]["params"]
    ).cuda()
        
        # Specify where to save checkpoints
    checkpoint_path = join(
        config["tensorboard_log_dir"],
        "lightning_checkpoints",
        f"{str(datetime.now()).replace(':', '-') + '-' + wandb.run.name + '-' + wandb.run.id}",
    )
    
    transformer.set_mlp_kwargs(mlp_kwargs)
    transformer.set_cfg(cfg)
    
    template_ginr_weights = train_dt.get_all_weights(0)[0]
    template_ginr = generate_mlp_from_weights(template_ginr_weights, mlp_kwargs, config=Config.config, isTemplate=True)
    transformer.set_template_ginr(template_ginr)
    
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
        #check_val_every_n_epoch=Config.get("val_fid_calculation_period"),
        check_val_every_n_epoch = 50,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        log_every_n_steps=1
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
    

def reconstruct_from_ginr_experiment_modulation(cfg: DictConfig):
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
    train_dt = ModulationDataset(
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
    mlp_from_ginr = generate_mlp_from_weights(ginr_weights, mlp_kwargs, config=Config.config)
    mlp_from_ginr.to(torch.device("cuda"))
    
    meshes = mlp_from_ginr.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="reconstruct_from_ginr_experiment")
    
    
    
def lin_interpolate_weight_space_experiment_modulation(cfg: DictConfig):
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
    train_dt = ModulationDataset(
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
    modulation_param_start = next(iterator_w)[0]
    modulation_param_end = next(iterator_w)[0]
    lin_erp_interval = 0.9
    modulation_param_lerp = lin_interpolate(modulation_param_start, modulation_param_end, lin_erp_interval)
    
    template_ginr_weights = train_dt.get_all_weights(0)[0]
    template_ginr = generate_mlp_from_weights(template_ginr_weights, mlp_kwargs, config=Config.config, isTemplate=True)
    
    specific_ginr_start = generate_mlp_from_weights(modulation_param_start, mlp_kwargs, config=Config.config, template_ginr=template_ginr)
    specific_ginr_start.to(torch.device("cuda"))
    
    specific_ginr_end = generate_mlp_from_weights(modulation_param_end, mlp_kwargs, config=Config.config, template_ginr=template_ginr)
    specific_ginr_end.to(torch.device("cuda"))
    
    meshes = specific_ginr_start.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="reconstruct_from_ginr_modulation_experiment_start")
    
    meshes = specific_ginr_end.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="reconstruct_from_ginr_modulation_experiment_end")
    
    specific_ginr_lerp = generate_mlp_from_weights(modulation_param_lerp, mlp_kwargs, config=Config.config, template_ginr=template_ginr)
    specific_ginr_lerp.to(torch.device("cuda"))
    
    meshes = specific_ginr_lerp.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="reconstruct_from_ginr_modulation_experiment_lerp=" + str(lin_erp_interval))
    
    pass

def effect_of_noise_experiment(cfg: DictConfig):
    # Define the noise level (scale factor)
    noise_level = 0.01
    
    # Sine wave parameters
    amplitude = 1.0  # Height of the wave
    w0 = 0.5  # Number of oscillations
    phase = 0        # Horizontal shift
    
    Config.config = config = cfg
    method = Config.get("method")
    print(method)
    
    # read mlp kwargs
    mlp_kwargs = Config.config["mlp_config"]["params"]
    
    #set train object names
    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )
    if not cfg.mlp_config.params.move:
        # COMMENTED FOR DEBUG PURPOSSES
        train_object_names = set([str.split(".")[0] for str in train_object_names])
        # SET TRAIN OBJECT NAMES MANUALLY FOR DEBUG PURPOSSES FOR SINGLE OBJECT
        #train_object_names = train_object_names.item().split(".")[0]
    
    # set train dataset
    train_dt = WeightDataset(
        "mlp_weights/Plane/single_3d_128_plane_multires_4_manifoldplus_slower_no_clipgrad",
        None, # wandb_logger,
        None, # model_dims
        mlp_kwargs,
        cfg,
        train_object_names,
        is_ginr=False
    )
    
    print(
        "Train dataset length: {}".format(
            len(train_dt)
        )
    )
    
    for w_i in train_dt:
        print(w_i[0].shape)
        print("Ziya stats:")
        print(f"min: {w_i[0].min()}, max: {w_i[0].max()}, mean: {w_i[0].mean()}, var {w_i[0].var()}, std: {w_i[0].std()}")
        break
    
    iterator_w = iter(train_dt)
    weights = next(iterator_w)[0]
    
    # Generate Sinusodial Noise
    # Generate a tensor of the same length as the vector
    t = torch.linspace(0, 1, steps=len(weights))
    # Generate the sine wave
    sine_wave = amplitude * torch.sin(w0 * t + phase)
    weights_sine = weights + sine_wave
    
    
    # Generate Gaussian noise
    gaussian_noise = torch.randn(weights.size())
    weights_gaussian = weights + gaussian_noise * noise_level
    
    
    # Generate combined noise
    combined_noise = sine_wave + gaussian_noise * noise_level
    weights_combined = weights + combined_noise
    
    
    config.mlps_type = "mlp3d"
    # generate mlp from weights  HyperDiff
    mlp = generate_mlp_from_weights(weights_sine, mlp_kwargs, config=config)
    reconstruct_from_mlp(mlp=mlp, cfg=cfg, additional_experiment_specifiers=["effect_of_noise_experiment", "sine"])
    
    config.mlps_type = "mlp3d"
    # generate mlp from weights  HyperDiff
    mlp = generate_mlp_from_weights(weights_gaussian, mlp_kwargs, config=config)
    reconstruct_from_mlp(mlp=mlp, cfg=cfg, additional_experiment_specifiers=["effect_of_noise_experiment", str(noise_level), "gaussian"])
    
    config.mlps_type = "mlp3d"
    # generate mlp from weights  HyperDiff
    mlp = generate_mlp_from_weights(weights_combined, mlp_kwargs, config=config)
    reconstruct_from_mlp(mlp=mlp, cfg=cfg, additional_experiment_specifiers=["effect_of_noise_experiment", str(noise_level), "combined"])
    
    
    ### GINR
    config.mlps_type = "ginr"
    # set train dataset
    train_dt = ModulationDataset(
        #"./mlp_weights/ginr_simplified_overfit_single",
        "./mlp_weights/ginr_simplified_overfit_relu",
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
        print("Modulation stats:")
        print(f"min: {w_i[0].min()}, max: {w_i[0].max()}, mean: {w_i[0].mean()}, var {w_i[0].var()}, std: {w_i[0].std()}")
        break
    
    iterator_w = iter(train_dt)
    modulation_param = next(iterator_w)[0]
    
    # Generate Sinusodial Noise
    # Generate a tensor of the same length as the vector
    t = torch.linspace(0, 1, steps=len(modulation_param))
    # Generate the sine wave
    sine_wave = amplitude * torch.sin(w0 * t + phase)
    # Apply sine noise
    modulation_param_sine = modulation_param + sine_wave
    
    
    # Generate Gaussian noise
    gaussian_noise = torch.randn(modulation_param.size())
    modulation_param_gaussian = modulation_param + gaussian_noise * noise_level
    
    # Generate combined noise
    combined_noise = sine_wave + gaussian_noise * noise_level
    modulation_param_combined = modulation_param + combined_noise
    
    # Template GINR
    template_ginr_weights = train_dt.get_all_weights(0)[0]
    template_ginr = generate_mlp_from_weights(template_ginr_weights, mlp_kwargs, config=Config.config, isTemplate=True)
    
    # Reconstruct gaussian
    specific_ginr = generate_mlp_from_weights(modulation_param_gaussian, mlp_kwargs, config=Config.config, template_ginr=template_ginr)
    specific_ginr.to(torch.device("cuda"))
    meshes = specific_ginr.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="effect_of_noise_ginr_simplified_gaussian_" + str(noise_level))
    
    # Reconstruct sine
    specific_ginr = generate_mlp_from_weights(modulation_param_sine, mlp_kwargs, config=Config.config, template_ginr=template_ginr)
    specific_ginr.to(torch.device("cuda"))
    meshes = specific_ginr.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="effect_of_noise_ginr_simplified_" + "sine")
    
    # Reconstruct combined
    specific_ginr = generate_mlp_from_weights(modulation_param_combined, mlp_kwargs, config=Config.config, template_ginr=template_ginr)
    specific_ginr.to(torch.device("cuda"))
    meshes = specific_ginr.overfit_one_shape(type='sdf')
    reconstruct_shape(meshes=meshes, epoch=-1, mode="effect_of_noise_ginr_simplified_" + "combined_" + str(noise_level))
    


if __name__ == "__main__":
    main()