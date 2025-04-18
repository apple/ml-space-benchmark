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
class Llama3_8b_Base:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_context_tokens: int = 8192
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = True
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@dataclass
class Llama3_70b_Base:
    model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    max_context_tokens: int = 8192
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = True
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@register_config("llama3_8b_qa")
@dataclass
class Llama3_8b_QA(Llama3_8b_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 2048
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("llama3_70b_qa")
@dataclass
class Llama3_70b_QA(Llama3_70b_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 2048
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("llama3_8b_dmtnav")
@dataclass
class Llama3_8b_DiscreteMapTextNav(Llama3_8b_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 1024
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("llama3_70b_dmtnav")
@dataclass
class Llama3_70b_DiscreteMapTextNav(Llama3_70b_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 1024
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("llama3_8b_cswm_text")
@dataclass
class Llama3_8b_CSWM_Text(Llama3_8b_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = -1
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("llama3_70b_cswm_text")
@dataclass
class Llama3_70b_CSWM_Text(Llama3_70b_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = -1
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("llama3_8b_mct_text")
@dataclass
class Llama3_8b_MCT_Text(Llama3_8b_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.8


@register_config("llama3_70b_mct_text")
@dataclass
class Llama3_70b_MCT_Text(Llama3_70b_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.8
