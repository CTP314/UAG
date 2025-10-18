# Run PushT

## With Docker

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/pusht/.venv
source examples/pusht/.venv/bin/activate
uv pip install -r examples/pusht/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```