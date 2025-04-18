# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any, Optional
import copy
from dataclasses import dataclass, field
from space.registry import register_config


VLLM_CFG = {
    "dtype": "auto",
    "trust_remote_code": True,
    "limit_mm_per_prompt": "image=20",
    "tensor_parallel_size": None,
}


@dataclass
class Phi35Vision_Base:
    model_name: str = "microsoft/Phi-3.5-vision-instruct"
    max_context_tokens: int = 128000
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = True
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@register_config("phi35vision_qa")
@dataclass
class Phi35Vision_QA(Phi35Vision_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1
