# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import glob
import json
import multiprocessing as mp
import os
from copy import deepcopy
from typing import Any

import fire
import imageio
import numpy as np
import tqdm

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


from space import get_config, get_agent, get_env
from space.utils.common import get_datetimestr
from space.utils.habitat import (
    compute_heading_from_quaternion,
    SPL,
    DistanceToGoal,
    Success,
)
from space.utils.visualizations import add_goal_to_obs
from space.utils.vllm_api import start_vllm_server

IMAGE_DOWNSCALING = 4
HABITAT_CONFIG = {
    "resolution": [512 * IMAGE_DOWNSCALING, 512 * IMAGE_DOWNSCALING],
    "forward_amount": 0.25,
    "turn_amount": 30,
}


def evaluate_on_env(
    agent_name: str,
    agent_cfg: dict[str, Any],
    env_dir: str,
    env_name: str,
    walkthrough_key: str,
    max_steps: int = 250,
):
    save_dir = agent_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    env = get_env(
        "NavEgoEnv",
        env_dir,
        habitat_kwargs=HABITAT_CONFIG,
        image_downscaling=IMAGE_DOWNSCALING,
    )

    # Load walkthrough from path
    walkthrough_video_frames = []
    with imageio.get_reader(os.path.join(env_dir, f"{walkthrough_key}.mp4")) as reader:
        for f in reader:
            walkthrough_video_frames.append(f)

    # Setup metrics
    task_info = env.get_task_info()
    d2g_metric = DistanceToGoal(env.sim, task_info["goal_position"])
    success_metric = Success(env.sim, task_info["goal_position"])
    spl_metric = SPL(env.sim, task_info["start_position"], task_info["goal_position"])

    # Setup agent
    agent = get_agent(agent_name, agent_cfg)
    agent.reset(walkthrough_key)

    # Provide walkthrough to agent
    agent.initialize_with_walkthrough(walkthrough_video_frames)

    # Provide goal to agent
    goal_desc = task_info["goal_desc"]
    goal_image = walkthrough_video_frames[-1]
    agent.initialize_with_goal(goal_desc, goal_image)

    # Start navigation
    # Initialize environment at walkthrough start state
    obs = env.reset()
    stop_issued = False
    pos, rot = env.get_sim_state()
    trajectory_positions = [pos]
    trajectory_rotations = [rot]
    actions_taken = []
    video_writer = imageio.get_writer(os.path.join(save_dir, "video.mp4"))
    vis_img = add_goal_to_obs(obs, goal_image)
    video_writer.append_data(vis_img)
    for _ in range(max_steps):
        act = agent.get_action(obs)
        if act not in ["move_forward", "turn_left", "turn_right", "stop"]:
            print(
                f"Obtained invalid action `{act}` from system. Replacing it with `turn_left`."
            )
            act = "turn_left"
        if act == "stop":
            stop_issued = True
            actions_taken.append(act)
            break
        obs = env.step(act)
        pos, rot = env.get_sim_state()
        trajectory_positions.append(pos)
        trajectory_rotations.append(rot)
        actions_taken.append(act)
        vis_img = add_goal_to_obs(obs, goal_image)
        video_writer.append_data(vis_img)
    video_writer.close()

    d2g = d2g_metric(trajectory_positions)
    success = success_metric(stop_issued, trajectory_positions)
    spl = spl_metric(stop_issued, trajectory_positions)
    metrics = {
        "distance_to_goal": d2g,
        "success": success,
        "spl": spl,
    }
    trajectory = {
        "positions": [t.tolist() for t in trajectory_positions],
        "headings": [compute_heading_from_quaternion(r) for r in trajectory_rotations],
        "actions": actions_taken,
    }

    env.close()

    # Calculate experiment cost
    eval_cost = agent.get_eval_cost()

    # Save information
    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(
            {
                "metrics": metrics,
                "trajectory": trajectory,
                "eval_cost": eval_cost,
            },
            fp,
        )

    return metrics, eval_cost, env_name


def _mp_helper(inputs: dict[str, Any]):
    return evaluate_on_env(**inputs)


def main(
    model_name: str,
    envs_dir: str,
    save_dir: str,
    walkthrough_key: str,
    max_steps: int = 250,
    n_workers: int = 8,
):
    agent_cfg = get_config(model_name)
    agent_name = agent_cfg["agent_name"]
    del agent_cfg["agent_name"]

    if agent_cfg["use_vllm"]:
        start_vllm_server(
            agent_cfg["model_name"], agent_cfg["host_port"], agent_cfg["vllm_cfg"]
        )

    save_dir = os.path.join(save_dir, model_name, get_datetimestr())
    os.makedirs(save_dir, exist_ok=True)

    # Load maze paths
    env_dirs = sorted(glob.glob(os.path.join(envs_dir, "*")))

    mp_inputs = []
    for env_dir in env_dirs:
        agent_cfg_m = deepcopy(agent_cfg)
        env_name = os.path.basename(env_dir)
        if "save_dir" in agent_cfg:
            agent_cfg_m["save_dir"] = os.path.join(save_dir, env_name)
        mp_inputs.append(
            {
                "agent_name": agent_name,
                "agent_cfg": agent_cfg_m,
                "env_dir": env_dir,
                "env_name": env_name,
                "walkthrough_key": walkthrough_key,
                "max_steps": max_steps,
            }
        )

    all_outputs = []
    pbar = tqdm.tqdm(total=len(mp_inputs), desc="Evaluating on SPACE navigation")
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for metrics, eval_cost, env_name in pool.imap_unordered(_mp_helper, mp_inputs):
            all_outputs.append(
                {"metrics": metrics, "eval_cost": eval_cost, "env_name": env_name}
            )
            pbar.update()
    pbar.close()

    all_outputs = sorted(all_outputs, key=lambda x: x["env_name"])
    all_metrics = []
    total_experiment_cost = {}
    for outputs in all_outputs:
        all_metrics.append(outputs["metrics"])
        for k, v in eval_cost.items():
            if k not in total_experiment_cost:
                total_experiment_cost[k] = 0.0
            total_experiment_cost[k] += v

    mean_metrics = {
        k: np.mean([m[k] for m in all_metrics]).item() for k in all_metrics[0].keys()
    }
    for k, v in mean_metrics.items():
        print(f"{k:20s}: {v:6.3f}")
    for k, v in total_experiment_cost.items():
        print(f"{k:20s}: {v:6.3f}")

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(
            {
                "all_metrics": all_metrics,
                "total_experiment_cost": total_experiment_cost,
            },
            fp,
        )


if __name__ == "__main__":
    fire.Fire(main)
