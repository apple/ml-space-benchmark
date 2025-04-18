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
    "config_format": "mistral",
    "load_format": "mistral",
    "tokenizer_mode": "mistral",
    "tensor_parallel_size": None,
}

PIXTRAL_VLLM_CFG = {
    "dtype": "auto",
    "trust_remote_code": True,
    "max_model_len": 32768,
    "enable_prefix_caching": True,
    "config_format": "mistral",
    "load_format": "mistral",
    "tokenizer_mode": "mistral",
    "limit_mm_per_prompt": "image=20",
    "tensor_parallel_size": None,
}


@dataclass
class Mixtral8x7b_Base:
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    max_context_tokens: int = 32768
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = False
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@dataclass
class Mixtral8x22b_Base:
    model_name: str = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    max_context_tokens: int = 32768
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = False
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@dataclass
class MistralLarge2407_Base:
    model_name: str = "mistralai/Mistral-Large-Instruct-2407"
    max_context_tokens: int = 128000
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = False
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(VLLM_CFG))


@dataclass
class Pixtral12b_Base:
    model_name: str = "mistralai/Pixtral-12B-2409"
    max_context_tokens: int = 32768
    completion_cost_per_mil: float = 0.0
    prompt_cost_per_mil: float = 0.0
    supports_system_prompt: bool = False
    use_vllm: bool = True
    vllm_cfg: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(PIXTRAL_VLLM_CFG)
    )


@register_config("mixtral8x7b_qa")
@dataclass
class Mixtral8x7b_QA(Mixtral8x7b_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("mixtral8x22b_qa")
@dataclass
class Mixtral8x22b_QA(Mixtral8x22b_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("mistral123b_qa")
@dataclass
class MistralLarge2407_QA(MistralLarge2407_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("pixtral12b_qa")
@dataclass
class Pixtral12b_QA(Pixtral12b_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: str = "8001"
    save_dir: Optional[str] = None
    subsampling_factor: int = 1


@register_config("mixtral8x7b_dmtnav")
@dataclass
class Mixtral8x7b_DiscreteMapTextNav(Mixtral8x7b_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("mixtral8x22b_dmtnav")
@dataclass
class Mixtral8x22b_DiscreteMapTextNav(Mixtral8x22b_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("mistral123b_dmtnav")
@dataclass
class MistralLarge2407_DiscreteMapTextNav(MistralLarge2407_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 4096
    max_history_length: int = 50
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("mixtral8x7b_cswm_text")
@dataclass
class Mixtral8x7b_CSWM_Text(Mixtral8x7b_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = -1
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("mixtral8x22b_cswm_text")
@dataclass
class Mixtral8x22b_CSWM_Text(Mixtral8x22b_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = -1
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("mistral123b_cswm_text")
@dataclass
class MistralLarge2407_CSWM_Text(MistralLarge2407_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = -1
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("mixtral8x7b_mct_text")
@dataclass
class Mixtral8x7b_MCT_Text(Mixtral8x7b_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("mixtral8x22b_mct_text")
@dataclass
class Mixtral8x22b_MCT_Text(Mixtral8x22b_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("mistral123b_mct_text")
@dataclass
class MistralLarge2407_MCT_Text(MistralLarge2407_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = 20
    host_port: str = "8001"
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
