import torch
from torch import nn
from torch import Tensor
import torchvision
import openpi.models.model as _model
from lerobot.common.policies.utils import (
    get_output_shape,
)
from lerobot.common.policies.diffusion.modeling_diffusion import (
    _replace_submodules,
    SpatialSoftmax,
)

class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, 
        vision_backbone: str,
        pretrained_backbone_weights: str | None,
        use_group_norm: bool,
        crop_shape: tuple[int, int] | None,
        spatial_softmax_num_keypoints: int,
    ):
        super().__init__()
        # Set up backbone.
        backbone_model = getattr(torchvision.models, vision_backbone)(
            weights=pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if use_group_norm:
            if pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = (3, *_model.IMAGE_RESOLUTION)
        dummy_shape_h_w = crop_shape if crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)
        self.feature_dim = spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x