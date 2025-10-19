import torch
import einops
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torchvision
from openpi.models_pytorch import diffusion_config
import openpi.models.model as _model
from lerobot.common.policies.utils import (
    get_dtype_from_parameters,
)
from lerobot.common.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    _make_noise_scheduler
)

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

from openpi.models_pytorch.encoder import DiffusionRgbEncoder

class DiffusionPolicy(nn.Module):
    def __init__(self, config: diffusion_config.DiffusionConfig):
        super().__init__()
        self.config = config
        global_cond_dim = config.action_dim
        
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(
                    config.vision_backbone,
                    config.pretrained_backbone_weights,
                    config.use_group_norm,
                    config.crop_shape,
                    config.spatial_softmax_num_keypoints,
                ) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(
                    config.vision_backbone,
                    config.pretrained_backbone_weights,
                    config.use_group_norm,
                    config.crop_shape,
                    config.spatial_softmax_num_keypoints,
                )
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        
        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps
       
    def to_bfloat16_for_selected_params(self, dtype):
        ...
       
    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, 
            train=train,
            image_keys=self.config.image_features,
            image_resolution=self.config.crop_shape if self.config.crop_shape is not None else _model.IMAGE_RESOLUTION
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
        
    def _prepare_global_conditioning(
        self,
        images: list[Tensor],
        img_masks,
        lang_tokens,
        lang_masks,
        state,
    ) -> Tensor:
        batch_size = state.shape[0]
        global_cond_feats = [state]
        images_tensor = torch.stack(images, dim=0)  # (num_cameras, B, C, H, W)

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = images_tensor
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera)
                    ]
                )
                
                img_features = einops.rearrange(
                    img_features_list, "(n b) ... -> b (n ...)", b=batch_size
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(
                        images_tensor, "n b ... -> (b n) ...", b=batch_size
                    )
                )
                img_features = einops.rearrange(
                    img_features, "(b n) ... -> b (n ...)", b=batch_size
                )
            global_cond_feats.append(img_features)
            
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
        
            
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        global_cond = self._prepare_global_conditioning(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        global_cond = global_cond.to(get_dtype_from_parameters(self.unet))
        trajectory = actions
        
        eps = torch.randn_like(trajectory) if noise is None else noise
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond)
        
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unknown prediction type {self.config.prediction_type}")
        
        loss = F.mse_loss(pred, target, reduction="none")
        
        if self.config.do_mask_loss_for_padding:
            raise NotImplementedError("Masking loss for padding is not implemented yet.")
        
        return loss.mean()

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            action_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(action_shape, device)
            
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        global_cond = self._prepare_global_conditioning(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        global_cond = global_cond.to(get_dtype_from_parameters(self.unet))
        x = self.conditional_sample(
            noisy_actions=noise,
            global_cond=global_cond,
            generator=torch.Generator(device=device)
        )
        return x
        
    def conditional_sample(
        self,
        noisy_actions: Tensor,
        global_cond: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        sample = noisy_actions
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, device=sample.device, dtype=torch.long),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        return sample