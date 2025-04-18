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

from space import get_config, get_agent, get_env
from space.utils.common import get_datetimestr
from space.envs.nav_dm import evaluate_path_efficiency
from space.utils.vllm_api import start_vllm_server


LOCAL_CONTEXT = 5


def evaluate_on_env(
    agent_name: str,
    agent_cfg: dict[str, Any],
    env_dir: str,
    obs_type: str,
    walkthrough_dir: str,
    env_name: str,
    walkthrough_key: str,
    max_steps: int = 100,
):
    save_dir = agent_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Setup environment
    env = get_env(
        "NavDiscreteMapEnv", os.path.join(env_dir, "info.json"), LOCAL_CONTEXT, obs_type
    )

    # Load walkthrough from path
    if obs_type == "image":
        walkthrough_obs = []
        with imageio.get_reader(
            os.path.join(walkthrough_dir, f"{walkthrough_key}_obs.mp4")
        ) as reader:
            for f in reader:
                walkthrough_obs.append(f)
    elif obs_type == "text":
        with open(
            os.path.join(walkthrough_dir, f"{walkthrough_key}_obs.json"), "r"
        ) as fp:
            walkthrough_obs = json.load(fp)
    else:
        raise ValueError(f"Observation type {obs_type} is not defined!")

    # Get ground-truth shortest path
    with open(os.path.join(walkthrough_dir, "shortestpath_info.json"), "r") as fp:
        gt_path = json.load(fp)["positions"]

    # Setup agent
    agent = get_agent(agent_name, agent_cfg)
    agent.reset(walkthrough_key)

    # Provide walkthrough to agent
    agent.initialize_with_walkthrough(walkthrough_obs)

    # Provide goal to agent
    goal_desc = f"landmark {env.goal_name}"
    agent.initialize_with_goal(goal_desc)

    # Start navigation
    # Initialize environment at walkthrough start state
    obs = env.reset()
    stop_issued = False
    pred_path = [env.current_location]
    actions_taken = []
    if obs_type == "image":
        writer = imageio.get_writer(os.path.join(save_dir, "observations.mp4"))
    else:
        writer = open(os.path.join(save_dir, "observations.txt"), "w")

    def _log_observation(obs):
        if obs_type == "image":
            writer.append_data(obs)
        else:
            writer.write(obs)
            writer.write("\n\n")
            writer.write("-" * 25)
            writer.write("\n\n")

    _log_observation(obs)
    for _ in range(max_steps):
        act = agent.get_action(obs)
        if act == "stop":
            stop_issued = True
            actions_taken.append(act)
            break
        obs = env.step(act)
        pred_path.append(env.current_location)
        actions_taken.append(act)
        _log_observation(obs)

    writer.close()

    # Calculate metrics
    ## evaluate_path_efficiency assumes (r, c) inputs for positions
    grid = np.array([[0 if col == "0" else 1 for col in row] for row in env.textmap])
    metrics = evaluate_path_efficiency(
        [(r, c) for c, r in gt_path],
        [(r, c) for c, r in pred_path],
        grid,
        stop_issued,
        dist_thresh=1.5,
    )

    # Calculate experiment cost
    eval_cost = agent.get_eval_cost()

    # Save information
    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(
            {
                "metrics": metrics,
                "pred_path": pred_path,
                "actions_taken": actions_taken,
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
    walkthroughs_dir: str,
    obs_type: str,
    save_dir: str,
    walkthrough_key: str,
    max_steps: int = 100,
    n_workers: int = 8,
):
    # Sanity checks
    assert obs_type in ["text", "image"]
    assert walkthrough_key in ["shortestpath", "walkthrough"]

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
                "obs_type": obs_type,
                "walkthrough_dir": os.path.join(walkthroughs_dir, env_name),
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
