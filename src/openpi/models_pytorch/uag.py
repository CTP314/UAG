import torch
import einops
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from openpi.models_pytorch import uag_config
from lerobot.common.policies.diffusion.modeling_diffusion import (
    _make_noise_scheduler
)

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

from openpi.models_pytorch.encoder import DiffusionRgbEncoder
from openpi.models_pytorch.recurrent_diffusion import GatedDeltaNetForRecurrentDiffusion
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetConfig

class UAGPolicy(nn.Module):
    def __init__(self, config: uag_config.UAGConfig):
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
        
        gated_deltanet_config = GatedDeltaNetConfig(
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
        )
        self.model = GatedDeltaNetForRecurrentDiffusion(
            gated_deltanet_config,
            input_dim=config.action_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            global_cond_dim=global_cond_dim,
        )
        
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