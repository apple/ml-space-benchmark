# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import glob
import json
import multiprocessing as mp
import os
from copy import deepcopy
from typing import Any

import fire
import numpy as np
import tqdm

from space import get_config, get_agent, get_env
from space.utils.common import get_datetimestr
from space.envs.mct import evaluate_path_efficiency
from space.utils.vllm_api import start_vllm_server


def evaluate_on_maze(
    agent_name: str,
    agent_cfg: dict[str, Any],
    maze_dir: str,
    maze_name: str,
    max_steps: int = 250,
):
    save_dir = agent_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Setup environment
    env = get_env("MCT_Env", maze_dir, agent_cfg["description_type"])

    # Setup agent
    agent = get_agent(agent_name, agent_cfg)
    agent.reset()

    # Get ground-truth shortest path
    gt_path = env.shortest_path
    obs, info = env.reset()
    pred_path = [env.current]
    actions = []
    issued_stop = False
    for _ in range(max_steps):
        act = agent.get_action(obs)
        if act not in ["up", "down", "left", "right", "stop"]:
            print(
                f"Obtained invalid action `{act}` from system. Replacing it with `up`."
            )
            act = "up"
        if act == "stop":
            actions.append(act)
            issued_stop = True
            break
        obs, info = env.step(act)
        pred_path.append(env.current)
        actions.append(act)

    # Calculate metrics
    metrics = evaluate_path_efficiency(
        gt_path, pred_path, env.maze, issued_stop=issued_stop
    )
    # Calculate experiment cost
    eval_cost = agent.get_eval_cost()

    return metrics, pred_path, actions, eval_cost, maze_name


def _mp_helper(inputs: dict[str, Any]):
    return evaluate_on_maze(**inputs)


def main(
    model_name: str,
    envs_dir: str,
    save_dir: str,
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
    maze_dirs = sorted(glob.glob(os.path.join(envs_dir, "*")))

    mp_inputs = []
    for maze_dir in maze_dirs:
        agent_cfg_m = deepcopy(agent_cfg)
        maze_name = os.path.basename(maze_dir)
        if "save_dir" in agent_cfg:
            agent_cfg_m["save_dir"] = os.path.join(save_dir, maze_name)
        mp_inputs.append(
            {
                "agent_name": agent_name,
                "agent_cfg": agent_cfg_m,
                "maze_dir": maze_dir,
                "maze_name": maze_name,
            }
        )

    all_outputs = []
    pbar = tqdm.tqdm(total=len(mp_inputs), desc="Evaluating on SPACE MCT")
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for (
            metrics,
            path_taken,
            actions,
            eval_cost,
            maze_name,
        ) in pool.imap_unordered(_mp_helper, mp_inputs):
            all_outputs.append(
                {
                    "metrics": metrics,
                    "path_taken": path_taken,
                    "actions": actions,
                    "eval_cost": eval_cost,
                    "maze_name": maze_name,
                }
            )
            # Save info for episode
            log_dir_i = os.path.join(save_dir, maze_name)
            with open(os.path.join(log_dir_i, "metrics.json"), "w") as fp:
                json.dump(
                    {
                        "metrics": metrics,
                        "actions": actions,
                        "path_taken": path_taken,
                    },
                    fp,
                )
            pbar.update()

    all_outputs = sorted(all_outputs, key=lambda x: x["maze_name"])
    all_metrics = []
    all_paths_taken = []
    all_actions = []
    total_experiment_cost = {}
    for outputs in all_outputs:
        all_metrics.append(outputs["metrics"])
        all_paths_taken.append(outputs["path_taken"])
        all_actions.append(outputs["actions"])
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
                "all_paths_taken": all_paths_taken,
                "all_actions": all_actions,
                "total_experiment_cost": total_experiment_cost,
            },
            fp,
        )


if __name__ == "__main__":
    fire.Fire(main)
