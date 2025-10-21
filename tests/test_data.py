import pickle
from openpi_client import image_tools

data_pkl_path = "/tmp/resize_debug.pkl"
data = pickle.load(open(data_pkl_path, "rb"))
print(data)
data["image"] = {k: image_tools.resize_with_pad(v, 96, 96) for k, v in data["image"].items()}
import ipdb; ipdb.set_trace()