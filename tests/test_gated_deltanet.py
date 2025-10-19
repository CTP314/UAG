import torch
from fla.models import GatedDeltaNetConfig, GatedDeltaNetModel

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
model = GatedDeltaNetModel(config)
model.eval()
model = model.to(device).to(dtype)

num_chunks = 4
chunk_size = T // num_chunks
input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, T)).to(device)
attention_mask = torch.ones((B, T), dtype=torch.bool).to(device)

output = model(input_ids=input_ids)
import ipdb; ipdb.set_trace()