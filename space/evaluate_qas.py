# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import json
import multiprocessing as mp
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any

import fire
import numpy as np
import tqdm

from space import get_config, get_agent
from space.utils.common import get_datetimestr
from space.utils.common import get_image_as_message, get_video_as_messages
from space.utils.vllm_api import start_vllm_server


def evaluate_on_qa(
    agent_name: str,
    agent_cfg: dict[str, Any],
    qa: dict[str, Any],
):
    # Setup agent
    agent = get_agent(agent_name, agent_cfg)
    agent.reset()
    question = qa["question"]
    answer = qa["answer"]
    if isinstance(question, list):
        question_content = []
        for q in question:
            if q.startswith("IMAGE:"):
                image_path = q[len("IMAGE:") :]
                message = get_image_as_message(
                    image_path=image_path,
                    model_name=agent.model_name,
                    image_detail=agent.image_detail,
                )
                question_content.append(message)
            elif q.startswith("VIDEO:"):
                video_path = q[len("VIDEO:") :]
                video_messages = get_video_as_messages(
                    video_path,
                    model_name=agent.model_name,
                    subsampling_factor=agent.subsampling_factor,
                    image_detail=agent.image_detail,
                )
                question_content.extend(video_messages)
            else:
                question_content.append(q)
    elif isinstance(question, str):
        question_content = question
    else:
        raise ValueError(
            f"Unable to parse question_content with type: {type(question)}"
        )

    P = agent.get_prediction(question_content, answer)
    metrics = {"accuracy": float(P == qa["answer"]) * 100.0}

    # Calculate experiment cost
    eval_cost = agent.get_eval_cost()

    return metrics, P, eval_cost


def _mp_helper(inputs: dict[str, Any]):
    return evaluate_on_qa(**inputs)


def main(
    model_name: str,
    data_path: str,
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

    with open(data_path, "r") as fp:
        dataset = json.load(fp)

    mp_inputs = []
    for i, qa in enumerate(dataset):
        agent_cfg_i = deepcopy(agent_cfg)
        if "save_dir" in agent_cfg_i:
            agent_cfg_i["save_dir"] = os.path.join(save_dir, f"qa_{i:05d}")
        mp_inputs.append(
            {
                "agent_name": agent_name,
                "agent_cfg": agent_cfg_i,
                "qa": qa,
            }
        )

    all_metrics = []
    all_predictions = []
    total_experiment_cost = {}

    pbar = tqdm.tqdm(total=len(mp_inputs), desc="Evaluating on QAs")
    with mp.Pool(n_workers, maxtasksperchild=1) as pool:
        for metrics, P, eval_cost in pool.imap(_mp_helper, mp_inputs):
            all_metrics.append(metrics)
            all_predictions.append(P)
            for k, v in eval_cost.items():
                if k not in total_experiment_cost:
                    total_experiment_cost[k] = 0.0
                total_experiment_cost[k] += v
            pbar.update()

    metrics = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            metrics[k].append(v)
    mean_metrics = {}
    for k, v_list in metrics.items():
        v_mean = np.mean(v_list).item()
        mean_metrics[k] = v_mean
        print(f"{k:<20s} | {v_mean:>7.3f}")

    with open(os.path.join(save_dir, "results.json"), "w") as fp:
        json.dump(
            {
                "all_metrics": all_metrics,
                "all_predictions": all_predictions,
                "total_experiment_cost": total_experiment_cost,
                "mean_metrics": mean_metrics,
            },
            fp,
        )


if __name__ == "__main__":
    fire.Fire(main)
