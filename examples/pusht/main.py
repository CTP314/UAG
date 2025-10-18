import pathlib
import uuid
import shutil
import gymnasium as gym
import gym_pusht
import collections
import tyro
import numpy as np
import dataclasses
import pickle
import time
import datetime
import imageio
import pandas as pd
import json
from openpi_client import websocket_client_policy as _websocket_client_policy

TASK = "Push the T-shaped block onto the T-shaped target."

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int | None = None
    
    seed: int = 7  # Random Seed (for reproducibility)
    run_id: str | None = None  # Run ID (for logging purposes)
    
    num_episode = 100  # Number of episodes to run
    save_video: bool = True  # Whether to save video of the episodes
    
    overwrite: bool = False
    resume: bool = False
    
def load_run_state(cache_dir: pathlib.Path):
    run_state_path = cache_dir / "run_state.pkl"
    with open(run_state_path, "rb") as f:
        run_state = pickle.load(f)
    return run_state["n"], run_state["np_state"]

def save_run_state(cache_dir: pathlib.Path, n: int, np_state: dict):
    run_state_path = cache_dir / "run_state.pkl"
    with open(run_state_path, "wb") as f:
        pickle.dump({"n": n, "np_state": np_state}, f)


def eval_pusht(args: Args):
    n, np_state = 0, None
    if args.run_id is None:
        current_time = datetime.datetime.now().strftime("%m%d_%H%M")
        args.run_id = f"{current_time}_{str(uuid.uuid4())[:4]}"
        log_dir = pathlib.Path(__file__).parent.resolve() / "logs" / args.run_id
        cache_dir = log_dir / ".cache"
    else:
        log_dir = pathlib.Path(__file__).parent.resolve() / "logs" / args.run_id
        cache_dir = log_dir / ".cache"
        if log_dir.exists():
            if args.overwrite:
                shutil.rmtree(log_dir)
            elif args.resume:
                try:
                    n, np_state = load_run_state(cache_dir)
                except ValueError:
                    raise ValueError(f"No run state found in {cache_dir} to resume from.")
            elif not args.resume:
                raise ValueError(f"Log directory {log_dir} already exists. Use --overwrite to overwrite or --resume to resume.")
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if np_state is not None:
        np.random.set_state(np_state)
    else:
        np.random.seed(args.seed)
    
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    meta_data = client.get_server_metadata()
    
    meta_file_path = log_dir / "meta.json"
    with open(meta_file_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    
    video_dir = log_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    record_file_path = log_dir / "results.csv"
    if not record_file_path.exists():
        df = pd.DataFrame(columns=["episode", "success", "elapsed_time", "episode_length"])
        df.to_csv(record_file_path, index=False)
    
    for episode in range(n, args.num_episode):
        observation, info = env.reset()
        action_plan = collections.deque()
        replay_images = []
        start_time = time.time()
        while True:
            if not action_plan:
                element = {
                    "observation/image": observation["pixels"],
                    "observation/state": observation["agent_pos"],
                    "prompt": TASK
                }
                try:
                    action_chunk = client.infer(element)["actions"]
                except Exception as e:
                    print(f"Error occurred while inferring actions: {e}")
                    exit(1)
                if args.replan_steps is not None:
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We requested {args.replan_steps} but got {len(action_chunk)}"
                    action_chunk = action_chunk[: args.replan_steps]
                action_plan.extend(action_chunk)
            
            action = action_plan.popleft()
            observation, reward, terminated, truncated, info = env.step(action)
            
            replay_images.append(observation["pixels"])
            
            if terminated or truncated:
                break
        
        success = info.get("is_success", False)
        print(f"Episode {episode + 1} {'succeeded' if success else 'failed'}.")    
        suffix = "success" if success else "failure"
        if args.save_video:
            imageio.mimwrite(
                video_dir / f"rollout_{episode + 1}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
                format="mp4",
            )
            
        elapsed_time = time.time() - start_time
        episode_length = len(replay_images)
        df = pd.read_csv(record_file_path)
        df = pd.concat([df, pd.DataFrame([{
            "episode": episode + 1,
            "success": success,
            "elapsed_time": elapsed_time,
            "episode_length": episode_length
        }])], ignore_index=True)
        df.to_csv(record_file_path, index=False)
            
        print(f"Episode {episode + 1}/{args.num_episode} completed. Cumulative success rate: {df['success'].mean()*100:.2f}%")
        save_run_state(cache_dir, episode + 1, np.random.get_state())

    # print summary information
    df = pd.read_csv(record_file_path)
    total_episodes = len(df)
    total_successes = df['success'].sum()
    overall_success_rate = (total_successes / total_episodes) * 100
    average_episode_length = df['episode_length'].mean()
    average_elapsed_time = df['elapsed_time'].mean()
    print("\nEvaluation Summary:")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {overall_success_rate:.2f}%")
    print(f"Average Episode Length: {average_episode_length:.2f} steps")
    print(f"Average Elapsed Time per Episode: {average_elapsed_time:.2f} seconds")
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    eval_pusht(args)