# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any, Optional
import copy
from dataclasses import dataclass, field
from space.registry import register_config


VLLM_CFG = {
    "dtype": "auto",
    "trust_remote_code": True,
    "enable_prefix_caching": True,
    "tensor_parallel_size": None,
}


@dataclass
class Yi15_9b_Base:
    model_name: str = "01-ai/Yi-1.5-9B-Chat-16K"
    max_context_tokens: int = 16000
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = True
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@dataclass
class Yi15_34b_Base:
    model_name: str = "01-ai/Yi-1.5-34B-Chat-16K"
    max_context_tokens: int = 16000
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = True
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@register_config("yi15_9b_qa")
@dataclass
class Yi15_9b_QA(Yi15_9b_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 2048
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("yi15_34b_qa")
@dataclass
class Yi15_34b_QA(Yi15_34b_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 2048
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("yi15_9b_dmtnav")
@dataclass
class Yi15_9b_DiscreteMapTextNav(Yi15_9b_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 1024
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("yi15_34b_dmtnav")
@dataclass
class Yi15_34b_DiscreteMapTextNav(Yi15_34b_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 1024
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("yi15_9b_cswm_text")
@dataclass
class Yi15_9b_CSWM_Text(Yi15_9b_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("yi15_34b_cswm_text")
@dataclass
class Yi15_34b_CSWM_Text(Yi15_34b_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("yi15_9b_mct_text")
@dataclass
class Yi15_9b_MCT_Text(Yi15_9b_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("yi15_34b_mct_text")
@dataclass
class Yi15_34b_MCT_Text(Yi15_34b_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
