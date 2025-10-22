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
from openpi.models_pytorch.recurrent_diffusion import GatedDeltaNetForRecurrentDiffusion, GatedDeltaNetConfig

def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype
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
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
        )
        self.model = GatedDeltaNetForRecurrentDiffusion(
            gated_deltanet_config,
            input_dim=config.action_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            global_cond_dim=global_cond_dim,
            use_linear_cond_proj=config.use_linear_cond_proj,
        )
        if self.config.dtype == "bfloat16":
            self.model.to(dtype=torch.bfloat16)
        
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
        
        image_tensor_bath_shape = images_tensor.shape[1:-3]
        images_tensor = images_tensor.reshape((images_tensor.shape[0], -1, *images_tensor.shape[-3:]))

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
            global_cond_feats.append(img_features.reshape((*image_tensor_bath_shape, -1)))
            
        return torch.cat(global_cond_feats, dim=-1)

    def _select_sparse_cond(
        self,
        indices: Tensor,  # [B, N]
        images: list[Tensor],  # List[[B, L, ...]]
        img_masks: list[Tensor] | None = None,  # List[[B, L]]
        lang_tokens: Tensor | None = None,  # [B, L, max_token_len]
        lang_masks: Tensor | None = None,  # [B, L, max_token_len]
        state: Tensor | None = None,  # [B, L, state_dim]
    ):
        selected_images = [
            torch.gather(
                img,
                dim=1,
                index=indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *img.shape[2:]),
            )
            for img in images
        ]
        selected_img_masks = None
        if img_masks is not None:
            selected_img_masks = [
                torch.gather(
                    img_mask,
                    dim=1,
                    index=indices.expand(-1, -1),
                )
                for img_mask in img_masks
            ]
        selected_lang_tokens = None
        if lang_tokens is not None:
            selected_lang_tokens = torch.gather(
                lang_tokens,
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, lang_tokens.shape[2]),
            )
        selected_lang_masks = None
        if lang_masks is not None:
            selected_lang_masks = torch.gather(
                lang_masks,
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, lang_masks.shape[2]),
            )
        selected_state = None
        if state is not None:
            selected_state = torch.gather(
                state,
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, state.shape[2]),
            )
        
        return selected_images, selected_img_masks, selected_lang_tokens, selected_lang_masks, selected_state

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        
        if self.config.cond_sample_mode == "sparse":
            global_cond_indices = torch.randint(
                0, min(actions.shape[1], self.config.max_cond_offset + 1),
                (actions.shape[0], 1),
                device=actions.device,
            )
            images, img_masks, lang_tokens, lang_masks, state = self._select_sparse_cond(
                global_cond_indices,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
            )
        else:
            global_cond_indices = None
        
        global_cond = self._prepare_global_conditioning(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        )
        global_cond = global_cond.to(get_dtype_from_parameters(self.model))
        trajectory = actions
        
        eps = torch.randn_like(trajectory) if noise is None else noise
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        if self.config.cond_sample_mode == "sparse":
            pred, _ = self.model(
                noisy_trajectory,
                timestep=timesteps[:, None].expand(-1, trajectory.shape[1]),
                global_cond=global_cond,
                global_cond_indices=global_cond_indices,
            )
        else:
            pred, _ = self.model.foward_dense(
                noisy_trajectory,
                timestep=timesteps[:, None].expand(-1, trajectory.shape[1]),
                global_cond=global_cond,
            )
        
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unknown prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target.to(get_dtype_from_parameters(self.model)), reduction="none")

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

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        global_cond = self._prepare_global_conditioning(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        global_cond = global_cond.to(get_dtype_from_parameters(self.model))
        x = self.sparse_conditional_sample(
            noisy_actions=noise,
            global_cond=global_cond[:, None],
            global_cond_indices=torch.zeros((bsize, 1), dtype=torch.long, device=device),
            generator=torch.Generator(device=device)
        )
        return x
    
    def sparse_conditional_sample(
        self,
        noisy_actions: Tensor,
        global_cond: Tensor | None,
        global_cond_indices: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        sample = noisy_actions
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output, _ = self.model(
                sample,
                torch.full(sample.shape[:2], t, device=sample.device, dtype=torch.long),
                global_cond=global_cond,
                global_cond_indices=global_cond_indices,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        return sample