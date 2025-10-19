import torch
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

L, B, T, H, D = 4, 4, 128, 4, 64
dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = GatedDeltaNetConfig(
    hidden_size=1024,
    num_kv_heads=1,
    num_heads=8,
    num_hidden_layers=18,
    intermediate_size=4096,
)
model = GatedDeltaNetForCausalLM(config)
# print params size
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
# change to MB unit
print("Model size (MB):", sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024)
model.eval()
model = model.to(device).to(dtype)

num_chunks = 4
chunk_size = T // num_chunks
input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T)).to(device)
attention_mask = torch.ones((B, T), dtype=torch.bool).to(device)
seq_start = torch.randint(low=1, high=chunk_size - 1, size=(B,)) * 0
attention_mask[torch.arange(T) < seq_start[:, None]] = False
print("seq_start:", seq_start)
ref = torch.cat([
    model(input_ids=input_ids[i:i+1, start:], use_cache=False).logits
    for i, start in enumerate(seq_start)
], dim=1)
print("ref shape:", ref.shape)

import time

start_time = time.time()

logits = []
out = model(
    input_ids=input_ids[:, :chunk_size],
    # attention_mask=attention_mask[:, :chunk_size],
    use_cache=True,
    past_key_values=None,
)

elapsed = time.time() - start_time
print(f"Time for first chunk: {elapsed:.4f} seconds")

logits, past_key_values = [out.logits], out.past_key_values
import pickle
for i in range(1, num_chunks):
    start, end = i * chunk_size, (i + 1) * chunk_size
    start_time = time.time()
    # for j in range(start, end):
    #     out = model(
    #         input_ids=input_ids[:, j:j+1],
    #         # attention_mask=attention_mask[:, :j+1],
    #         use_cache=True,
    #         past_key_values=past_key_values,
    #     )
    #     logits.append(out.logits)
    #     past_key_values = out.past_key_values
    #     pickle.dump(past_key_values, open(f"data/transformer/past_key_values_{i}_{j}.pkl", "wb"))
    out = model(
        input_ids=input_ids[:, start:end],
        # attention_mask=attention_mask[:, :end],
        use_cache=True,
        past_key_values=past_key_values,
    )
    logits.extend([out.logits[:, k:k+1] for k in range(out.logits.shape[1])])
    past_key_values = out.past_key_values
    elapsed = time.time() - start_time
    print(f"Time for chunk {i}: {elapsed:.4f} seconds")
    
gen = torch.cat(logits, 1)
gen = torch.cat([gen[i:i+1, start:] for i, start in enumerate(seq_start)], 1)

# compare gen and ref
mse = torch.mean((gen - ref) ** 2).item()
print("MSE between generation and reference:", mse)