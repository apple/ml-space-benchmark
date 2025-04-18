# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Optional

from dataclasses import dataclass
from space.registry import register_config


@dataclass
class GPT4V_Base:
    model_name: str = "gpt-4-turbo-2024-04-09"
    max_context_tokens: int = 128000
    completion_cost_per_mil: float = 30.0
    prompt_cost_per_mil: float = 10.0
    supports_system_prompt: bool = True
    use_vllm: bool = False


@dataclass
class GPT4O_Base:
    model_name: str = "gpt-4o-2024-05-13"
    max_context_tokens: int = 128000
    completion_cost_per_mil: float = 15.0
    prompt_cost_per_mil: float = 5.0
    supports_system_prompt: bool = True
    use_vllm: bool = False


@register_config("gpt4v_qa")
@dataclass
class GPT4V_QA(GPT4V_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    subsampling_factor: int = 1


@register_config("gpt4o_qa")
@dataclass
class GPT4O_QA(GPT4O_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    subsampling_factor: int = 1


@register_config("gpt4v_egonav")
@dataclass
class GPT4V_EgoNav(GPT4V_Base):
    agent_name: str = "EgoNav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("gpt4o_egonav")
@dataclass
class GPT4O_EgoNav(GPT4V_Base):
    agent_name: str = "EgoNav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("gpt4v_dminav")
@dataclass
class GPT4V_DiscreteMapImageNav(GPT4V_Base):
    agent_name: str = "DiscreteMapImage_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("gpt4o_dminav")
@dataclass
class GPT4O_DiscreteMapImageNav(GPT4O_Base):
    agent_name: str = "DiscreteMapImage_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("gpt4v_dmtnav")
@dataclass
class GPT4V_DiscreteMapTextNav(GPT4V_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("gpt4o_dmtnav")
@dataclass
class GPT4O_DiscreteMapTextNav(GPT4O_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("gpt4v_cswm_vision")
@dataclass
class GPT4V_CSWM_Vision(GPT4V_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "vision"
    max_new_tokens: int = 4096
    max_history_length: int = -1
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9


@register_config("gpt4v_cswm_text")
@dataclass
class GPT4V_CSWM_Text(GPT4V_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = -1
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9


@register_config("gpt4o_cswm_vision")
@dataclass
class GPT4O_CSWM_Vision(GPT4O_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "vision"
    max_new_tokens: int = 4096
    max_history_length: int = -1
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9


@register_config("gpt4o_cswm_text")
@dataclass
class GPT4O_CSWM_Text(GPT4O_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = -1
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9


@register_config("gpt4v_mct_vision")
@dataclass
class GPT4V_MCT_Vision(GPT4V_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "image"
    max_new_tokens: int = 4096
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "high"
    context_truncation_factor: float = 0.9


@register_config("gpt4v_mct_text")
@dataclass
class GPT4V_MCT_Text(GPT4V_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9


@register_config("gpt4o_mct_vision")
@dataclass
class GPT4O_MCT_Vision(GPT4O_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "image"
    max_new_tokens: int = 4096
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "high"
    context_truncation_factor: float = 0.9


@register_config("gpt4o_mct_text")
@dataclass
class GPT4O_MCT_Text(GPT4O_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
