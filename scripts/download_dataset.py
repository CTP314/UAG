import tyro
import dataclasses
from datasets import load_dataset

@dataclasses.dataclass
class Args:
    dataset_name: str = "lerobot/pusht"

args = tyro.cli(Args)
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset(args.dataset_name)