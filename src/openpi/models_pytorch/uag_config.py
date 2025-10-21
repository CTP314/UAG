import dataclasses
import jax
import jax.numpy as jnp
from typing_extensions import override
from openpi.models import model as _model
import openpi.shared.array_typing as at
import logging
import safetensors.torch

logger = logging.getLogger("openpi")

@dataclasses.dataclass(frozen=True)
class UAGConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    action_horizon: int = 16
    action_dim: int = 32
    max_token_len: int = 48
    
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    image_features: tuple[str, ...] = (
        "base_0_rgb",
    )
    
    # num_hidden_layers: int = 18
    # num_heads: int = 4
    # head_dim: int = 256
    # hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_heads: int = 4
    head_dim: int = 128
    hidden_size: int = 1024
    
    diffusion_step_embed_dim: int = 128
    
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    
    # Inference
    num_inference_steps: int | None = None

    # Training Strategy
    cond_sample_mode: str = "sparse"  # "full" or "sparse"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.UAG
    
    def create(self, rng: at.KeyArrayLike):
        raise NotImplementedError("UAG is only support for pytorch")
    
    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_shape = self.crop_shape if self.crop_shape is not None else _model.IMAGE_RESOLUTION
        image_spec = jax.ShapeDtypeStruct([batch_size, *image_shape, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_specW
    
    def load_pytorch(self, train_config, weight_path: str):
        from openpi.models_pytorch import diffusion
        logger.info(f"train_config: {train_config}")
        model = diffusion.DiffusionPolicy(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model