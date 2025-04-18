# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any

import subprocess as sp
import time
import torch
import openai
from openai import OpenAI


def is_server_ready(model_name: str, host_port: str, vllm_cfg: dict[str, Any]):
    try:
        client = OpenAI(base_url=f"http://localhost:{host_port}/v1", api_key="EMPTY")
        models_available = [i.id for i in client.models.list().data]
        if model_name in models_available:
            return True
        else:
            return False
    except openai.APIConnectionError:
        return False


def start_vllm_server(model_name: str, host_port: str, vllm_cfg: dict[str, Any]):
    # Check if server is already available
    if is_server_ready(model_name, host_port, vllm_cfg):
        return

    # Host server
    if "tensor_parallel_size" in vllm_cfg and vllm_cfg["tensor_parallel_size"] is None:
        vllm_cfg["tensor_parallel_size"] = torch.cuda.device_count()
    command = ["vllm", "serve", model_name, "--port", host_port]
    for k, v in vllm_cfg.items():
        if k in ["enable_prefix_caching", "trust_remote_code"] and v:
            command.append(f"--{k}")
        else:
            command.extend([f"--{k}", f"{v}"])
    sp.Popen(command, stderr=sp.DEVNULL, stdout=sp.DEVNULL)

    # Wait till vllm server is ready
    while not is_server_ready(model_name, host_port, vllm_cfg):
        print(
            f"Model {model_name} not yet available on vLLM. Waiting for server to be up..."
        )
        time.sleep(5)
