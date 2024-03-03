import os

from dataset import VoxelDataset, WeightDataset, ModulationDataset
from hd_utils import Config, get_mlp
from hyperdiffusion import HyperDiffusion

# Using it to make pyrender work on clusters
os.environ["PYOPENGL_PLATFORM"] = "egl"
import sys
from datetime import datetime
from os.path import join
from model import create_model

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import ldm.ldm.modules.diffusionmodules.openaimodel
import wandb
from transformer import Transformer

sys.path.append("siren")


@hydra.main(
    version_base=None,
    config_path="configs/diffusion_configs",
    config_name="train_plane_ginr_simplified_relu",
)
def main(cfg: DictConfig):
    Config.config = config = cfg
    method = Config.get("method")
    mlp_kwargs = None

    # In HyperDiffusion, we need to know the specifications of MLPs that are used for overfitting
    if "hyper" in method:
        mlp_kwargs = Config.config["mlp_config"]["params"]

    wandb.init(
        project="hyperdiffusion_ginr_modulation_debug",
        dir=config["tensorboard_log_dir"],
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        tags=[Config.get("mode")],
        mode="disabled" if Config.get("disable_wandb") else "online",
        config=dict(config),
    )

    wandb_logger = WandbLogger()
    wandb_logger.log_text("config", ["config"], [[str(config)]])
    print("wandb", wandb.run.name, wandb.run.id)

    train_dt = val_dt = test_dt = None

    # Although it says train, it includes all the shapes but we only extract training ones in WeightDataset
    mlps_folder_train = Config.get("mlps_folder_train")

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
    # Initialize UNet for Voxel baseline
    else:
        model = ldm.ldm.modules.diffusionmodules.openaimodel.UNetModel(
            **Config.config["unet_config"]["params"]
        ).float()


    dataset_path = os.path.join(Config.config["dataset_dir"], Config.config["dataset"])
    train_object_names = np.genfromtxt(
        os.path.join(dataset_path, "train_split.lst"), dtype="str"
    )
    
    train_object_names = set([str.split(".")[0] for str in train_object_names])
    # SET TRAIN OBJECT NAMES MANUALLY FOR DEBUG PURPOSSES FOR SINGLE OBJECT
    #train_object_names = train_object_names.item().split(".")[0]
    
    # Check if dataset folder already has train,test,val split; create otherwise.
    if method == "hyper_3d":
        mlps_folder_all = mlps_folder_train
        all_object_names = np.array(
            [obj for obj in os.listdir(dataset_path) if ".lst" not in obj]
        )
        total_size = len(all_object_names)
        val_size = int(total_size * 0.05)
        test_size = int(total_size * 0.15)
        train_size = total_size - val_size - test_size
        if not os.path.exists(os.path.join(dataset_path, "train_split.lst")):
            train_idx = np.random.choice(
                total_size, train_size + val_size, replace=False
            )
            test_idx = set(range(total_size)).difference(train_idx)
            val_idx = set(np.random.choice(train_idx, val_size, replace=False))
            train_idx = set(train_idx).difference(val_idx)
            print(
                "Generating new partition",
                len(train_idx),
                train_size,
                len(val_idx),
                val_size,
                len(test_idx),
                test_size,
            )

            # Sanity checking the train, val and test splits
            assert len(train_idx.intersection(val_idx.intersection(test_idx))) == 0
            assert len(train_idx.union(val_idx.union(test_idx))) == total_size
            assert (
                len(train_idx) == train_size
                and len(val_idx) == val_size
                and len(test_idx) == test_size
            )

            np.savetxt(
                os.path.join(dataset_path, "train_split.lst"),
                all_object_names[list(train_idx)],
                delimiter=" ",
                fmt="%s",
            )
            np.savetxt(
                os.path.join(dataset_path, "val_split.lst"),
                all_object_names[list(val_idx)],
                delimiter=" ",
                fmt="%s",
            )
            np.savetxt(
                os.path.join(dataset_path, "test_split.lst"),
                all_object_names[list(test_idx)],
                delimiter=" ",
                fmt="%s",
            )

        val_object_names = np.genfromtxt(
            os.path.join(dataset_path, "val_split.lst"), dtype="str"
        )
        val_object_names = set([str.split(".")[0] for str in val_object_names])
        test_object_names = np.genfromtxt(
            os.path.join(dataset_path, "test_split.lst"), dtype="str"
        )
        test_object_names = set([str.split(".")[0] for str in test_object_names])

        train_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
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
        val_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            val_object_names,
        )
        test_dt = WeightDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            test_object_names,
        )
    elif "ginr_modulated" in method:
        val_object_names = np.genfromtxt(
            os.path.join(dataset_path, "val_split.lst"), dtype="str"
        )
        val_object_names = set([str.split(".")[0] for str in val_object_names])
        #val_object_names = val_object_names.item().split(".")[0]
        
        test_object_names = np.genfromtxt(
            os.path.join(dataset_path, "test_split.lst"), dtype="str"
        )
        test_object_names = set([str.split(".")[0] for str in test_object_names])
        #test_object_names = test_object_names.item().split(".")[0]
        
        train_dt = ModulationDataset(
            mlps_folder_train,
            wandb_logger,
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
        
        val_dt = ModulationDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            val_object_names,
            is_ginr= True if Config.get("mlps_type") == "ginr" else False
        )
        
        test_dt = ModulationDataset(
            mlps_folder_train,
            wandb_logger,
            model.dims,
            mlp_kwargs,
            cfg,
            test_object_names,
            is_ginr= True if Config.get("mlps_type") == "ginr" else False
        )

    # These two dl's are just placeholders, during val and test evaluation we are looking at test_split.lst,
    # val_split.lst files, inside calc_metrics methods
    val_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]), batch_size=1, shuffle=False
    )
    test_dl = DataLoader(
        torch.utils.data.Subset(train_dt, [0]), batch_size=1, shuffle=False
    )

    print(
        "Train dataset length: {} Val dataset length: {} Test dataset length {}".format(
            len(train_dt), len(val_dt), len(test_dt)
        )
    )

        
    
    input_data = next(iter(train_dl))[0]
    print(
        "Input data shape, min, max:",
        input_data.shape,
        input_data.min(),
        input_data.max(),
    )

    best_model_save_path = Config.get("best_model_save_path")
    model_resume_path = Config.get("model_resume_path")

    # Initialize HyperDiffusion
    diffuser = HyperDiffusion(
        model, train_dt, val_dt, test_dt, mlp_kwargs, input_data.shape, method, cfg
    )
    
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

    # Specify where to save checkpoints
    checkpoint_path = join(
        config["tensorboard_log_dir"],
        "lightning_checkpoints",
        f"{str(datetime.now()).replace(':', '-') + '-' + wandb.run.name + '-' + wandb.run.id}",
    )
    best_acc_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/1-NN-CD-acc",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-nn-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    )

    best_mmd_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val/lgan_mmd-CD",
        mode="min",
        dirpath=checkpoint_path,
        filename="best-val-mmd-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
    )

    last_model_saver = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="last-{epoch:02d}-{train_loss:.2f}-{val_fid:.2f}",
        save_on_train_epoch_end=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    
    checkpoint_callback = ModelCheckpoint(dirpath='./save', save_last=True)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=Config.get("epochs"),
        strategy="ddp",
        logger=wandb_logger,
        default_root_dir=checkpoint_path,
        callbacks=[
            #best_acc_checkpoint,
            #best_mmd_checkpoint,
            #last_model_saver,
            lr_monitor,
            #checkpoint_callback
        ],
        check_val_every_n_epoch=Config.get("val_fid_calculation_period"),
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        log_every_n_steps=1
    )
    
    

    if Config.get("mode") == "train":
        # If model_resume_path is provided (i.e., not None), the training will continue from that checkpoint
        trainer.fit(diffuser, train_dl, val_dl, ckpt_path=model_resume_path)
        pass

    # best_model_save_path is the path to saved best model
    trainer.test(
        diffuser,
        test_dl,
        ckpt_path=best_model_save_path if Config.get("mode") == "test" else None,
    )
    wandb_logger.finalize("Success")


if __name__ == "__main__":
    main()
