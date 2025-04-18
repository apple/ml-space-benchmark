# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import glob
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any

import fire
import imageio
import numpy as np
import tqdm

from space import get_config, get_agent, get_env
from space.utils.common import get_datetimestr
from space.utils.vllm_api import start_vllm_server

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def play_game(
    agent_name: str,
    agent_cfg: dict[str, Any],
    game_dir: str,
    game_name: str,
    game_mode: str,
):
    save_dir = agent_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Setup environment
    env_name = "Vision_CSWM_Env" if game_mode == "vision" else "Text_CSWM_Env"
    env = get_env(env_name, game_dir)

    # Setup agent
    agent = get_agent(agent_name, agent_cfg)
    agent.reset()

    obs = env.reset()
    n_boxes = len(env.rects)
    # Setup logging
    os.makedirs(save_dir, exist_ok=True)
    # Begin evaluation
    n_steps_taken = 0
    actions = []
    actions_true = []
    observations = [obs]
    for _ in range(env.max_steps):
        act = agent.get_action(obs)
        if not env.is_valid_action(act):
            act = env.sample_random_action()
        act_true = env.apply_r2t_mapping(act)
        obs, _, done, _ = env.step(act)
        n_steps_taken += 1
        observations.append(obs)
        actions.append(act)
        actions_true.append(act_true)
        if done:
            break
    if game_mode == "vision":
        with imageio.get_writer(
            os.path.join(save_dir, "video.mp4"), fps=2
        ) as video_writer:
            for obs in observations:
                video_writer.append_data(obs)
    elif game_mode == "text":
        with open(os.path.join(save_dir, "game_log.json"), "w") as fp:
            json.dump(observations, fp)
    else:
        raise ValueError(f"Undefined game mode: {game_mode}")

    n_treasures_found = env.n_collected
    success = float(env.n_collected == len(env.treasures)) * 100.0
    metrics = {
        "success": success,
        "n_steps_taken": n_steps_taken,
        "n_treasures_found": n_treasures_found,
    }

    with open(os.path.join(save_dir, "info.json"), "w") as fp:
        json.dump(
            {
                "actions": actions,
                "actions_true": actions_true,
                "metrics": metrics,
                "n_boxes": n_boxes,
            },
            fp,
        )

    # Calculate experiment cost
    eval_cost = agent.get_eval_cost()

    return actions, actions_true, metrics, eval_cost, game_name, n_boxes


def _mp_helper(inputs: dict[str, Any]):
    return play_game(**inputs)


def main(
    model_name: str,
    envs_dir: str,
    save_dir: str,
    n_workers: int = 8,
    game_mode: str = "vision",
):
    # Sanity checks
    assert game_mode in ["vision", "text"]

    agent_cfg = get_config(model_name)
    agent_name = agent_cfg["agent_name"]
    del agent_cfg["agent_name"]

    if agent_cfg["use_vllm"]:
        start_vllm_server(
            agent_cfg["model_name"], agent_cfg["host_port"], agent_cfg["vllm_cfg"]
        )

    save_dir = os.path.join(save_dir, model_name, get_datetimestr())
    os.makedirs(save_dir, exist_ok=True)

    # Load game paths
    game_dirs = sorted(glob.glob(os.path.join(envs_dir, "*")))

    mp_inputs = []
    for game_dir in game_dirs:
        agent_cfg_m = deepcopy(agent_cfg)
        game_name = os.path.basename(game_dir)
        agent_cfg_m["save_dir"] = os.path.join(save_dir, game_name)
        mp_inputs.append(
            {
                "agent_name": agent_name,
                "agent_cfg": agent_cfg_m,
                "game_dir": game_dir,
                "game_name": game_name,
                "game_mode": game_mode,
            }
        )

    all_outputs = []
    pbar = tqdm.tqdm(total=len(mp_inputs), desc="Evaluating on Space CSWM task")
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for (
            actions,
            actions_true,
            metrics,
            eval_cost,
            game_name,
            n_boxes,
        ) in pool.imap_unordered(_mp_helper, mp_inputs):
            all_outputs.append(
                {
                    "actions": actions,
                    "actions_true": actions_true,
                    "metrics": metrics,
                    "eval_cost": eval_cost,
                    "game_name": game_name,
                    "n_boxes": n_boxes,
                }
            )
            pbar.update()

    all_outputs = sorted(all_outputs, key=lambda x: x["game_name"])
    all_metrics = defaultdict(list)
    all_actions = []
    all_actions_true = []
    all_n_boxes = []
    total_experiment_cost = defaultdict(int)
    for output in all_outputs:
        for k, v in output["metrics"].items():
            all_metrics[k].append(v)
        all_actions.append(output["actions"])
        all_actions_true.append(output["actions_true"])
        all_n_boxes.append(output["n_boxes"])
        for k, v in output["eval_cost"].items():
            total_experiment_cost[k] += v

    mean_metrics = {k: np.mean(v).item() for k, v in all_metrics.items()}
    for k, v in mean_metrics.items():
        print(f"{k:20s}: {v:6.3f}")
    for k, v in total_experiment_cost.items():
        print(f"{k:20s}: {v:6.3f}")

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(
            {
                "all_metrics": all_metrics,
                "mean_metrics": mean_metrics,
                "all_actions": all_actions,
                "all_actions_true": all_actions_true,
                "all_n_boxes": all_n_boxes,
                "total_experiment_cost": total_experiment_cost,
            },
            fp,
        )


if __name__ == "__main__":
    fire.Fire(main)
