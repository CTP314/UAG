import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if len(image.shape) == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "b c h w -> b h w c")
    return image

@dataclasses.dataclass(frozen=True)
class PushTInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    
    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        state = data["observation/state"]
        has_expanded = False
        if len(base_image.shape) == 3:
            base_image = np.expand_dims(base_image, axis=0)
            state = np.expand_dims(state, axis=0)
            has_expanded = True
            
        empty_image_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": np.asarray(state).astype(np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),  # Placeholder for missing left wrist image
                "right_wrist_0_rgb": np.zeros_like(base_image),  # Placeholder for missing right wrist image
            },
            "prompt": data["prompt"],
            "image_mask": {
                "base_0_rgb": np.ones(base_image.shape[0], dtype=bool),
                "left_wrist_0_rgb": np.full(base_image.shape[0], empty_image_mask, dtype=bool),
                "right_wrist_0_rgb": np.full(base_image.shape[0], empty_image_mask, dtype=bool),
            }
        }
        
        if "actions" in data:
            inputs["actions"] = data["actions"]
            
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        if has_expanded:
            for key in inputs:
                if isinstance(inputs[key], dict):
                    for subkey in inputs[key]:
                        inputs[key][subkey] = np.squeeze(inputs[key][subkey], axis=0)
                else:
                    inputs[key] = np.squeeze(inputs[key], axis=0)
            
        return inputs
    
@dataclasses.dataclass(frozen=True)
class PushTOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :2])}