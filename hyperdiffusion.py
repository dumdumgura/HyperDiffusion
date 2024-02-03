import copy
import os
from operator import add

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import wandb
from diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)
from evaluation_metrics_3d import compute_all_metrics, compute_all_metrics_4d
from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights,
                      render_mesh, render_meshes, reconstruct_shape)
from siren import sdf_meshing
from siren.dataio import anime_read
from siren.experiment_scripts.test_sdf import SDFDecoder


class HyperDiffusion(pl.LightningModule):
    def __init__(
        self, model, train_dt, val_dt, test_dt, mlp_kwargs, image_shape, method, cfg
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.method = method
        self.mlp_kwargs = mlp_kwargs
        self.val_dt = val_dt
        self.train_dt = train_dt
        self.test_dt = test_dt
        self.ae_model = None
        self.sample_count = min(
            8, Config.get("batch_size")
        )  # it shouldn't be more than 36 limited by batch_size
        fake_data = torch.randn(*image_shape)

        encoded_outs = fake_data
        print("encoded_outs.shape", encoded_outs.shape)
        timesteps = Config.config["timesteps"]
        # original betas
        betas = torch.tensor(np.linspace(1e-4, 2e-2, timesteps))
        # beta trial
        self.image_size = encoded_outs[:1].shape

        # Initialize diffusion utiities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType[cfg.diff_config.params.model_mean_type],
            model_var_type=ModelVarType[cfg.diff_config.params.model_var_type],
            loss_type=LossType[cfg.diff_config.params.loss_type],
            diff_pl_module=self,
        )
        if "ginr_modulated" in self.method:
            self.template_ginr_weights = self.train_dt.get_all_weights(0)[0]
            self.template_ginr = generate_mlp_from_weights(self.template_ginr_weights, self.mlp_kwargs, config=Config.config, isTemplate=True)
            
        if self.cfg.normalize_input:
            self.data_mean = None
            self.data_std = None

    def forward(self, images):
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(images.shape[0],))
            .long()
            .to(self.device)
        )
        images = images * self.cfg.normalization_factor
        x_t, e = self.diff.q_sample(images, t)
        x_t = x_t.float()
        e = e.float()
        return self.model(x_t, t), e

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=Config.get("lr"))
        if self.cfg.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.cfg.scheduler_step, gamma=0.9
            )
            return [optimizer], [scheduler]
        return optimizer

    def grid_to_mesh(self, grid):
        grid = np.where(grid > 0, True, False)
        vox_grid = trimesh.voxel.VoxelGrid(grid)
        try:
            vox_grid = vox_grid.marching_cubes
        except:
            return vox_grid.as_boxes()
        vert = vox_grid.vertices
        if len(vert) == 0:
            return vox_grid
        vert /= grid.shape[-1]
        vert = 2 * vert - 1
        vox_grid.vertices = vert
        return vox_grid

    def training_step(self, train_batch, batch_idx):
        # Extract input_data (either voxel or weight) which is the first element    of the tuple
        input_data = train_batch[0]

        # At the first step output first element in the dataset as a sanit check
        if "hyper" in self.method and self.trainer.global_step == 0:
            curr_weights = Config.get("curr_weights")
            img = input_data[0].flatten()[:curr_weights]
            print(f"img_shape: {img.shape}")
            if "ginr_modulated" in self.method:
                mlp = generate_mlp_from_weights(img, self.mlp_kwargs, config=Config.config, template_ginr=self.template_ginr)
                mlp.to(torch.device("cuda"))
                mesh = mlp.overfit_one_shape(type='sdf', level=0)
                reconstruct_shape(meshes=mesh, epoch=-1, mode="first_mesh_hyperdiffusion")
                mesh = trimesh.Trimesh(mesh[0]['vertices'], mesh[0]['faces'])
                out_imgs = render_meshes([mesh])
                self.logger.log_image(
                    "sanity_check_renders", out_imgs, step=self.current_epoch
                )
            else:
                mlp = generate_mlp_from_weights(img, self.mlp_kwargs, config=Config.config)
                sdf_decoder = SDFDecoder(
                    self.mlp_kwargs.model_type,
                    None,
                    "nerf" if self.mlp_kwargs.model_type == "nerf" else "mlp",
                    self.mlp_kwargs,
                )
                sdf_decoder.model = mlp.cuda()
                if not self.mlp_kwargs.move:
                    sdf_meshing.create_mesh(
                        sdf_decoder,
                        "meshes/first_mesh",
                        N=128,
                        level=0.5 if self.mlp_kwargs.output_type == "occ" else 0,
                    )

                print("Input images shape:", input_data.shape)
                
        # Output statistics every 100 step
        if self.trainer.global_step % 100 == 0:
            print(f"input.data.shape = {input_data.shape}")
            print(
                "Orig weights[0].stats",
                input_data.min().item(),
                input_data.max().item(),
                input_data.mean().item(),
                input_data.std().item(),
                input_data.var().item(),
            )

        
        #### STARTS HERE
        # Sample a diffusion timestep
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(input_data.shape[0],))
            .long()
            .to(self.device)
        )

        # Normalize
        if self.cfg.normalize_input:
            input_data = (input_data - self.data_mean.cuda()) / self.data_std.cuda()
            
        # Execute a diffusion forward pass
        loss_terms = self.diff.training_losses(
            self.model,
            input_data * self.cfg.normalization_factor,
            t,
            self.mlp_kwargs,
            self.logger,
            model_kwargs=None,
        )
        loss_mse = loss_terms["loss"].mean()
        self.logger.log_metrics({"train/train_loss": loss_mse}, step=self.global_step)

        loss = loss_mse
        return loss

    def validation_step(self, val_batch, batch_idx):
        metric_fn = self.calc_metrics
        
        metrics = metric_fn("train")
        for metric_name in metrics:
            self.logger.log_metrics({"train/" + metric_name: metrics[metric_name]}, step=self.global_step)
        metrics = metric_fn("val")
        for metric_name in metrics:
            self.logger.log_metrics({"val/" + metric_name: metrics[metric_name]}, step=self.global_step)
            
        if self.cfg.vis_intermediate_timesteps:
            sample_x_0s, indices = self.visualize_timestep_outputs()
            
            sample_x_0s = torch.vstack(sample_x_0s)
            meshes, _ = self.generate_meshes(sample_x_0s)
            out_imgs = render_meshes(meshes)
            self.logger.log_image(
                "intermediate_timestep_renders",
                out_imgs,
                step=self.current_epoch,
                caption=list(map(add, ["t="]*len(indices), map(str, indices)))
            )
            

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        epoch_loss = sum(output["loss"] for output in outputs) / len(outputs)
        self.logger.log_metrics({"epoch_loss": epoch_loss}, step=self.current_epoch)

        # Handle 3D sample generation
        if "hyper" in self.method:
            if self.current_epoch % Config.get("vis_calculation_period") == 0:
                
                x_0s = (
                    self.diff.ddim_sample_loop(self.model, (self.cfg.num_of_generations, *self.image_size[1:]))
                    .cpu()
                    .float()
                )
                
                ## Denormalization
                if self.cfg.normalize_input:
                    x_0s = (x_0s * self.data_std.cpu()) + self.data_mean.cpu()
                    
                x_0s = x_0s / self.cfg.normalization_factor
                print(
                    "x_0s[0].stats",
                    x_0s.min().item(),
                    x_0s.max().item(),
                    x_0s.mean().item(),
                    x_0s.std().item(),
                    x_0s.var().item()
                )
                
                meshes, sdfs = self.generate_meshes(x_0s, None, res=256, level=0)
                if sdfs:
                    print(
                        "sdfs.stats",
                        sdfs.min().item(),
                        sdfs.max().item(),
                        sdfs.mean().item(),
                        sdfs.std().item(),
                    )
                out_imgs = render_meshes(meshes)
                self.logger.log_image(
                    "generated_renders", out_imgs, step=self.current_epoch,   caption= \
                        list(map(add, [f"epoch="+str(self.current_epoch)]*self.cfg.num_of_generations, map(add, [" sample="]*self.cfg.num_of_generations, map(str, range(self.cfg.num_of_generations)))))
                )

    def generate_meshes(self, x_0s, folder_name="meshes", info="0", res=256, level=0.07):
        x_0s = x_0s.view(len(x_0s), -1)
        curr_weights = Config.get("curr_weights")
        x_0s = x_0s[:, :curr_weights]
        meshes = []
        sdfs = []
        for i, weights in enumerate(x_0s):
            if "ginr_modulated" in self.method:
                mlp = generate_mlp_from_weights(weights, self.mlp_kwargs, config=Config.config, template_ginr=self.template_ginr)
                mlp.to(torch.device("cuda"))
                mesh = mlp.overfit_one_shape(type='sdf', res=res, level=level)
                mesh = trimesh.Trimesh(mesh[0]['vertices'], mesh[0]['faces'])
                meshes.append(mesh)
                continue
            mlp = generate_mlp_from_weights(weights, self.mlp_kwargs)
            sdf_decoder = SDFDecoder(
                self.mlp_kwargs.model_type,
                None,
                "nerf" if self.mlp_kwargs.model_type == "nerf" else "mlp",
                self.mlp_kwargs,
            )
            sdf_decoder.model = mlp.cuda().eval()
            with torch.no_grad():
                effective_file_name = (
                    f"{folder_name}/mesh_epoch_{self.current_epoch}_{i}_{info}"
                    if folder_name is not None
                    else None
                )
                if self.mlp_kwargs.move:
                    for i in range(16):
                        v, f, sdf = sdf_meshing.create_mesh(
                            sdf_decoder,
                            effective_file_name,
                            N=res,
                            level=0
                            if self.mlp_kwargs.output_type in ["occ", "logits"]
                            else 0,
                            time_val=i,
                        )  # 0.9
                        if (
                            "occ" in self.mlp_kwargs.output_type
                            or "logits" in self.mlp_kwargs.output_type
                        ):
                            tmp = copy.deepcopy(f[:, 1])
                            f[:, 1] = f[:, 2]
                            f[:, 2] = tmp
                        sdfs.append(sdf)
                        mesh = trimesh.Trimesh(v, f)
                        meshes.append(mesh)
                else:
                    v, f, sdf = sdf_meshing.create_mesh(
                        sdf_decoder,
                        effective_file_name,
                        N=res,
                        level=level
                        if self.mlp_kwargs.output_type in ["occ", "logits"]
                        else 0,
                    )
                    if (
                        "occ" in self.mlp_kwargs.output_type
                        or "logits" in self.mlp_kwargs.output_type
                    ):
                        tmp = copy.deepcopy(f[:, 1])
                        f[:, 1] = f[:, 2]
                        f[:, 2] = tmp
                    sdfs.append(sdf)
                    mesh = trimesh.Trimesh(v, f)
                    meshes.append(mesh)
        #sdfs = torch.stack(sdfs)
        return meshes, sdfs

    def print_summary(self, flat, func):
        var = func(flat, dim=0)
        print(
            var.shape,
            var.mean().item(),
            var.std().item(),
            var.min().item(),
            var.max().item(),
        )
        print(var.shape, func(flat))

    def calc_metrics(self, split_type):   
        dataset_path = os.path.join(
            Config.config["dataset_dir"],
            Config.config["dataset"] + f"_{self.cfg.val.num_points}_pc",
        )
        test_object_names = np.genfromtxt(
            os.path.join(dataset_path, f"{split_type}_split.lst"), dtype="str"
        )
        print("test_object_names.length", len(test_object_names))

        orig_meshes_dir = f"orig_meshes/run_{wandb.run.name}"
        os.makedirs(orig_meshes_dir, exist_ok=True)

        # During validation, only use some of the val and train shapes for speed
        if split_type == "val" and self.cfg.val.num_samples is not None:
            test_object_names = test_object_names[: self.cfg.val.num_samples]
        elif split_type == "train" and self.cfg.val.num_samples is not None:
            test_object_names = test_object_names[: self.cfg.val.num_samples]
        n_points = self.cfg.val.num_points

        # First process ground truth shapes
        pcs = []
        for obj_name in test_object_names:
            pc = np.load(os.path.join(dataset_path, obj_name + ".npy"))
            pc = pc[:, :3]

            pc = torch.tensor(pc).float()
            if split_type == "test":
                pc = pc.float()
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
                pc = (pc - shift) / scale
            pcs.append(pc)
        r = Rotation.from_euler("x", 90, degrees=True)
        self.logger.experiment.log({"3d_gt": wandb.Object3D(r.apply(np.array(pcs[0])))})
        ref_pcs = torch.stack(pcs)

        # We are generating slightly more than ref_pcs
        number_of_samples_to_generate = int(len(ref_pcs) * self.cfg.test_sample_mult)

        # Then process generated shapes
        sample_x_0s = []
        test_batch_size = 100 if "hyper_3d" in self.cfg.method else self.cfg.batch_size

        for _ in tqdm(range(number_of_samples_to_generate // test_batch_size)):
            # burada listedeki her randoomdan samplellama icin denormalization yapmamiz lazim
            x_0s = self.diff.ddim_sample_loop(
                self.model,
                (
                    test_batch_size, *self.image_size[1:]
                ),
            )
            ## Denormalization
            if self.cfg.normalize_input:
                x_0s = (x_0s * self.data_std.cpu()) + self.data_mean.cpu()
                    
            sample_x_0s.append(x_0s)
            
            

        if number_of_samples_to_generate % test_batch_size != 0:
            x_0s = self.diff.ddim_sample_loop(
                self.model,
                (
                    number_of_samples_to_generate % test_batch_size,
                    *self.image_size[1:],
                ),
            )
            
            ## Denormalization
            if self.cfg.normalize_input:
                x_0s = (x_0s * self.data_std.cpu()) + self.data_mean.cpu()
                    
            sample_x_0s.append(x_0s)
            

        sample_x_0s = torch.vstack(sample_x_0s)
        torch.save(sample_x_0s, f"{orig_meshes_dir}/prev_sample_x_0s.pth")
        print(sample_x_0s.shape)
        if self.cfg.dedup:
            sample_dist = torch.cdist(
                sample_x_0s,
                sample_x_0s,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            sample_dist_min = sample_dist.kthvalue(k=2, dim=1)[0]
            sample_dist_min_sorted = torch.argsort(sample_dist_min, descending=True)[
                : int(len(ref_pcs) * 1.01)
            ]
            sample_x_0s = sample_x_0s[sample_dist_min_sorted]
            print(
                "sample_dist.shape, sample_x_0s.shape",
                sample_dist.shape,
                sample_x_0s.shape,
            )
        torch.save(sample_x_0s, f"{orig_meshes_dir}/sample_x_0s.pth")
        print("Sampled")

        print("Running marching cubes")
        sample_batch = []
        for x_0s in tqdm(sample_x_0s):
            if "hyper" in self.cfg.method:
                mesh, _ = self.generate_meshes(
                    x_0s.unsqueeze(0) / self.cfg.normalization_factor,
                    None,
                    res=64 if split_type == "test" else 256,
                    #level=1.386 if split_type == "test" else 0,
                )
                mesh = mesh[0]
            else:
                grid = np.where(x_0s[0].cpu() > 0, True, False)
                vox_grid = trimesh.voxel.VoxelGrid(grid)
                if np.any(grid):
                    vox_mesh = vox_grid.marching_cubes
                    vert = vox_mesh.vertices
                    vert = vert - np.mean(vert, axis=0, keepdims=True)
                    v_max = np.amax(vert)
                    v_min = np.amin(vert)
                    vert *= 0.95 / (max(abs(v_min), abs(v_max)))
                    vox_mesh.vertices = vert
                else:
                    vox_mesh = vox_grid.as_boxes()
                mesh = vox_mesh
            if len(mesh.vertices) > 0:
                pc = torch.tensor(mesh.sample(n_points))
                if not self.cfg.mlp_config.params.move and "hyper" in self.cfg.method:
                    pc = pc * 2
                pc = pc.float()
                if split_type == "test":
                    pc = pc.float()
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                    pc = (pc - shift) / scale
            else:
                print("Empty mesh")
                if split_type in ["val", "train"]:
                    pc = torch.zeros_like(ref_pcs[0])
                else:
                    continue
            sample_batch.append(pc)
        print("Marching cubes completed")

        print("number of samples generated:", len(sample_batch))
        sample_batch = sample_batch[: len(ref_pcs)]
        print("number of samples generated (after clipping):", len(sample_batch))
        sample_pcs = torch.stack(sample_batch)
        assert len(sample_pcs) == len(ref_pcs)
        torch.save(sample_pcs, f"{orig_meshes_dir}/samples.pth")

        self.logger.experiment.log(
            {"3d_gen": wandb.Object3D(r.apply(np.array(sample_pcs[0])))}
        )
        print("Starting metric computation for", split_type)

        fid = calculate_fid_3d(
            sample_pcs.to(self.device), ref_pcs.to(self.device), self.logger
        )
        metrics = compute_all_metrics(
            sample_pcs.to(self.device),
            ref_pcs.to(self.device),
            16 if split_type == "test" else 16,
            self.logger,
        )
        metrics["fid"] = fid.item()

        print("Completed metric computation for", split_type)

        return metrics

    def test_step(self, *args, **kwargs):
        if self.cfg.calculate_metric_on_test:
            metric_fn = (
                self.calc_metrics_4d
                if self.cfg.mlp_config.params.move
                else self.calc_metrics
            )
            metrics = metric_fn("test")
            print("test", metrics)
            for metric_name in metrics:
                self.logger.log_metrics({"test/" + metric_name: metrics[metric_name]}, step=self.global_step)

        # If it's HyperDiffusion, let's calculate some statistics on training dataset
        if "hyper_3d" in self.method:
            x_0s = []
            for i, img in enumerate(self.train_dt):
                x_0s.append(img[0])
            x_0s = torch.stack(x_0s).to(self.device)
            flat = x_0s.view(len(x_0s), -1)
            # return
            print(x_0s.shape, flat.shape)
            print("Variance With zero-padding")
            self.print_summary(flat, torch.var)
            print("Variance Without zero-padding")
            self.print_summary(flat[:, : Config.get("curr_weights")], torch.var)

            print("Mean With zero-padding")
            self.print_summary(flat, torch.mean)
            print("Mean Without zero-padding")
            self.print_summary(flat[:, : Config.get("curr_weights")], torch.mean)

            stdev = x_0s.flatten().std(unbiased=True).item()
            oai_coeff = (
                0.538 / stdev
            )  # 0.538 is the variance of ImageNet pixels scaled to [-1, 1]
            print(f"Standard Deviation: {stdev}")
            print(f"OpenAI Coefficient: {oai_coeff}")

            # Then, sampling some new shapes -> outputting and rendering them
            x_0s = self.diff.ddim_sample_loop(
                self.model, (16, *self.image_size[1:]), clip_denoised=False
            )
            
            ## Denormalization
            if self.cfg.normalize_input:
                print("self.data_mean: ", self.data_mean, "\tself.data_std:", self.data_std)
                x_0s = (x_0s * self.data_std.cpu()) + self.data_mean.cpu()
            
            x_0s = x_0s / self.cfg.normalization_factor

            print(
                "x_0s[0].stats",
                x_0s.min().item(),
                x_0s.max().item(),
                x_0s.mean().item(),
                x_0s.std().item(),
            )
            #out_pc_imgs = []

            # Handle 3D generation
            out_imgs = []
                
            meshes, _ = self.generate_meshes(x_0s)
            out_imgs = render_meshes(meshes)

            self.logger.log_image(
                "generated_renders_test", out_imgs, step=self.current_epoch
            )
            #self.logger.log_image(
            #    "generated_renders_pc_test", out_pc_imgs, step=self.current_epoch
            #)
        
        
    def visualize_timestep_outputs(self):
        sample_x_0s = []
        indices = []
        for i, sample in enumerate(self.diff.ddim_sample_loop_progressive(
            self.model,
            shape=(1, *self.image_size[1:])
        )):
            if (i+1) % 125 == 0:
                x_0s = sample["sample"].cpu().float()
                
                ## Denormalization
                if self.cfg.normalize_input:
                    x_0s = (x_0s * self.data_std.cpu()) + self.data_mean.cpu()
            
                sample_x_0s.append(x_0s)
                indices.append(i)
        
        return sample_x_0s, indices