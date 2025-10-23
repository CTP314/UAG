import openpi.models_pytorch.recurrent_diffusion as _recurrent_diffusion
from fla.models.gated_deltanet import GatedDeltaNetConfig
import torch
import time
import timeit

def test_recurrent_diffusion_inference_speed(
    model: _recurrent_diffusion.GatedDeltaNetForRecurrentDiffusion,
    B: int, L: int, I: int, C: int, device: torch.device, dtype: torch.dtype
):
    x = torch.randn(B, L, I, device=device, dtype=dtype)
    diffusion_steps = torch.randint(0, 1000, (B, L), device=device)
    # global_cond = torch.randn(B, L, C, device=device, dtype=dtype)
    # global_cond_mask = torch.ones(B, 1, device=device, dtype=dtype)
    global_cond_sparse = torch.randn(B, 1, C, device=device, dtype=dtype)
    global_cond_indices = torch.randint(0, L, (B, 1), device=device)
    global_cond_mask = torch.zeros(B, L, device=device, dtype=torch.bool)
    global_cond = torch.zeros(B, L, C, device=device, dtype=dtype)
    for b in range(B):
        for i in range(global_cond_indices.shape[1]):
            idx = global_cond_indices[b, i]
            global_cond_mask[b, idx] = True
            global_cond[b, idx, :] = global_cond_sparse[b, i, :]
    model.eval()
    with torch.no_grad():
        # Warm up
        start_time = time.perf_counter()
        num_warmup_iters = 100
        for _ in range(num_warmup_iters):
            _ = model(
                x,
                timestep=diffusion_steps,
                global_cond=global_cond_sparse,
                global_cond_indices=global_cond_indices,
            )
        end_time = time.perf_counter()
        warmup_time = end_time - start_time
        avg_warmup_time = warmup_time / num_warmup_iters
        print(f"Average warm-up time per iteration: {avg_warmup_time * 1000:.2f} ms")
        # Measure time
        num_iters = 50
        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = model(
                x,
                timestep=diffusion_steps,
                global_cond=global_cond_sparse,
                global_cond_indices=global_cond_indices,
            )
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_iter = total_time / num_iters
        print(f"Average sparse inference time per iteration: {avg_time_per_iter * 1000:.2f} ms")
        
        # Measure full sequence inference time
        start_time = time.perf_counter()
        num_iters = 50
        for _ in range(num_iters):
            _ = model.forward_dense(
                x,
                timestep=diffusion_steps,
                global_cond=global_cond,
                global_cond_mask=global_cond_mask,
            )
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_iter = total_time / num_iters
        print(f"Average inference time per iteration: {avg_time_per_iter * 1000:.2f} ms")
        
def test_recurrent_diffusion():
    config = GatedDeltaNetConfig(
        num_hidden_layers=3,
        num_heads=8,
        # num_hidden_layers=18,
        # num_heads=4,
        # head_dim=256,
        # hidden_size=1024,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    input_dim, diffusion_step_embed_dim, global_cond_dim = 32, 128, 128
    B, L = 1, 128
    model = _recurrent_diffusion.GatedDeltaNetForRecurrentDiffusion(
        config=config,
        input_dim=input_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        global_cond_dim=global_cond_dim,
    )
    model.to(device=device, dtype=dtype)
    # print model
    # print(model)
    # print number of parameters in Billions
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Number of parameters: {num_params:.3f}B")
    
    test_recurrent_diffusion_inference_speed(
        model=model,
        B=B,
        L=L,
        I=input_dim,
        C=global_cond_dim,
        device=device,
        dtype=dtype,
    )
    
def test_recurrent_diffusion_film():
    config = GatedDeltaNetConfig(
        num_hidden_layers=3,
        num_heads=8,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    input_dim, diffusion_step_embed_dim, global_cond_dim = 32, 128, 128
    model = _recurrent_diffusion.GatedDeltaNetForRecurrentDiffusionFiLM(
        config=config,
        input_dim=input_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        global_cond_dim=global_cond_dim,
    )
    B, L = 1, 128
    model.to(device=device, dtype=dtype)
    # print model
    # print(model)
    test_recurrent_diffusion_inference_speed(
        model=model,
        B=B,
        L=L,
        I=input_dim,
        C=global_cond_dim,
        device=device,
        dtype=dtype,
    )
    
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionConfig
import lerobot.configs.types as _types

def test_unet_diffusion_inference_speed(
    model: DiffusionConditionalUnet1d,
    B: int, L: int, I: int, C: int, device: torch.device, dtype: torch.dtype
):
    x = torch.randn(B, L, I, device=device, dtype=dtype)
    diffusion_steps = torch.randint(0, 1000, (B,), device=device)
    global_cond = torch.randn(B, C, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        # Warm up
        start_time = time.perf_counter()
        num_warmup_iters = 100
        for _ in range(num_warmup_iters):
            _ = model(
                x,
                timestep=diffusion_steps,
                global_cond=global_cond,
            )
        end_time = time.perf_counter()
        warmup_time = end_time - start_time
        avg_warmup_time = warmup_time / num_warmup_iters
        print(f"Average warm-up time per iteration: {avg_warmup_time * 1000:.2f} ms")
        # Measure time
        num_iters = 50
        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = model(
                x,
                timestep=diffusion_steps,
                global_cond=global_cond,
            )
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_iter = total_time / num_iters
        print(f"Average inference time per iteration: {avg_time_per_iter * 1000:.2f} ms")

def test_unet_diffusion():
    input_dim, global_cond_dim = 32, 128
    config = DiffusionConfig(
        output_features={_types.FeatureType.ACTION: _types.PolicyFeature(
            type=_types.FeatureType.ACTION,
            shape=(input_dim,),
        )},
        diffusion_step_embed_dim=128,
    )
    print(config.action_feature)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model = DiffusionConditionalUnet1d(
        config=config,
        global_cond_dim=global_cond_dim,
    ).to(device=device)
    # print model
    # print(model)
    # print number of parameters in Billions
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Number of parameters: {num_params:.3f}B")

    test_unet_diffusion_inference_speed(
        model=model,
        B=1,
        L=128,
        I=input_dim,
        C=global_cond_dim,
        device=device,
        dtype=dtype,
    )

if __name__ == "__main__":
    # test_unet_diffusion()
    test_recurrent_diffusion()
    test_recurrent_diffusion_film()