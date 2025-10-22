import torch
from torch import nn
from torch import Tensor
import math
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetPreTrainedModel, GatedDeltaNetConfig, GatedDeltaNetBlock
from fla.models.utils import Cache
from fla.modules.layernorm import RMSNorm
from typing import List

class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class DiffusionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.Mish(),
            nn.Linear(input_dim * 4, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(dtype=self.net[0].weight.dtype)
        return self.net(x)

class GatedDeltaNetForRecurrentDiffusion(GatedDeltaNetPreTrainedModel):
    def __init__(self, 
        config: GatedDeltaNetConfig,
        input_dim: int,
        diffusion_step_embed_dim: int,
        global_cond_dim: int = 0,
        use_linear_cond_proj: bool = True,
    ):
        super().__init__(config)
        self.config = config
        
        self.diffusion_step_encoder = self.make_step_encoder(diffusion_step_embed_dim)
        self.cond_dim = global_cond_dim + diffusion_step_embed_dim
        self.uncond_dim = diffusion_step_embed_dim
        self.input_proj = DiffusionMLP(input_dim, config.hidden_size)
        
        # self.hidden_cond_proj = DiffusionMLP(config.hidden_size + self.cond_dim, config.hidden_size)
        # self.hidden_uncond_proj = DiffusionMLP(config.hidden_size + self.uncond_dim, config.hidden_size)
        if use_linear_cond_proj:
            self.hidden_cond_projs = nn.ModuleList(
                nn.Linear(config.hidden_size + self.cond_dim, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            )
            self.hidden_uncond_projs = nn.ModuleList(
                nn.Linear(config.hidden_size + self.uncond_dim, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            )
        else:
            self.hidden_cond_projs = nn.ModuleList(
                DiffusionMLP(config.hidden_size + self.cond_dim, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            )
            self.hidden_uncond_projs = nn.ModuleList(
                DiffusionMLP(config.hidden_size + self.uncond_dim, config.hidden_size)
                for _ in range(config.num_hidden_layers)
            )

        self.layers = nn.ModuleList([GatedDeltaNetBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        
        self.gradient_checkpointing = False
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.output_head = nn.Linear(config.hidden_size, input_dim)
        
        self.post_init()

    def make_step_encoder(self, diffusion_step_embed_dim: int) -> nn.Module:
        return nn.Sequential(
            DiffusionSinusoidalPosEmb(diffusion_step_embed_dim),
            DiffusionMLP(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )
    
    def forward_dense(
        self,
        x: Tensor, # [B, L, input_dim]
        timestep: Tensor, # [B, L]
        global_cond: Tensor | None = None, # [B, L, global_cond_dim]
        global_cond_mask: Tensor | None = None,  # [B, L]
        *,
        attention_mask: Tensor | None = None,
        past_key_values: Cache | List[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
    ):
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        
        # x: (batch_size, seq_len, input_dim)
        # timestep: (batch_size, seq_len)
        # global_cond: (batch_size, seq_len, global_cond_dim)
        # global_cond_mask: (batch_size, seq_len)

        timesteps_emb = self.diffusion_step_encoder(timestep)
        hidden_states = self.input_proj(x)  # (batch_size, seq_len, hidden_size)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)
        
        if global_cond_mask is not None:
            global_cond_mask = global_cond_mask.unsqueeze(-1).to(hidden_states.dtype)

        for layer_idx, layer in enumerate(self.layers):
            if global_cond_mask is None and global_cond is not None:
                hidden_states_cond_input = torch.cat([
                    hidden_states,
                    timesteps_emb,
                    global_cond
                ], dim=-1)
                hidden_states = self.hidden_cond_projs[layer_idx](hidden_states_cond_input)
            else:
                hidden_states_uncond_input = torch.cat([
                    hidden_states,
                    timesteps_emb
                ], dim=-1)
                hidden_states_uncond_output = self.hidden_uncond_projs[layer_idx](hidden_states_uncond_input)
                if global_cond is not None:
                    assert global_cond_mask is not None, "global_cond_mask must be provided when global_cond is used"
                    hidden_states_cond_input = torch.cat([
                        hidden_states,
                        timesteps_emb,
                        global_cond
                    ], dim=-1)
                    hidden_states_cond_output = self.hidden_cond_projs[layer_idx](hidden_states_cond_input)
                    hidden_states = global_cond_mask * hidden_states_cond_output + (1 - global_cond_mask) * hidden_states_uncond_output
                else:
                    hidden_states = hidden_states_uncond_output

            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
        hidden_states = self.norm(hidden_states)
        output = self.output_head(hidden_states)
        
        return output, past_key_values
    
    def forward(
        self,
        x: Tensor, # [B, L, input_dim]
        timestep: Tensor, # [B, L]
        global_cond: Tensor | None = None, # [B, N, global_cond_dim]
        global_cond_indices: Tensor | None = None, # [B, N]
        *,
        attention_mask: Tensor | None = None,
        past_key_values: Cache | List[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
    ):
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        
        timesteps_emb = self.diffusion_step_encoder(timestep)
        hidden_states = self.input_proj(x)  # (batch_size, seq_len, hidden_size)
        
        B, L, H = hidden_states.shape
        _, _, H_t = timesteps_emb.shape

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)
            
        for layer_idx, layer in enumerate(self.layers):
            hidden_states_uncond_input = torch.cat([
                hidden_states,
                timesteps_emb
            ], dim=-1)
            hidden_states_uncond_output = self.hidden_uncond_projs[layer_idx](hidden_states_uncond_input)
            
            if global_cond is not None:
                assert global_cond_indices is not None, "global_cond_indices must be provided when global_cond is used"
                indices_h = global_cond_indices.unsqueeze(-1).expand(-1, -1, H)  # (B, N, H)
                indices_t = global_cond_indices.unsqueeze(-1).expand(-1, -1, H_t)  # (B, N, H_t)

                cond_hidden_states = torch.gather(hidden_states, 1, indices_h)  # (B, N, global_cond_dim)
                cond_timesteps_emb = torch.gather(timesteps_emb, 1, indices_t)  # (B, N, diffusion_step_embed_dim)
                
                hidden_states_cond_input = torch.cat([
                    cond_hidden_states,
                    cond_timesteps_emb,
                    global_cond
                ], dim=-1)

                hidden_states_cond_output = self.hidden_cond_projs[layer_idx](hidden_states_cond_input)
                
                hidden_states = torch.scatter(
                    hidden_states_uncond_output,
                    1,
                    indices_h,
                    hidden_states_cond_output,
                )
            else:
                hidden_states = hidden_states_uncond_output
                
            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
        hidden_states = self.norm(hidden_states)
        output = self.output_head(hidden_states)
        return output, past_key_values