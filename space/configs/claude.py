# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Optional
from dataclasses import dataclass

from space.registry import register_config


@dataclass
class Claude35Sonnet_Base:
    model_name: str = "claude-3-5-sonnet-20240620"
    max_context_tokens: int = 200000
    completion_cost_per_mil: float = 15.0
    prompt_cost_per_mil: float = 3.0
    supports_system_prompt: bool = False
    max_images_per_query: int = 10
    use_vllm: bool = False


@register_config("claude35sonnet_qa")
@dataclass
class Claude35Sonnet_QA(Claude35Sonnet_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    subsampling_factor: int = 1


@register_config("claude35sonnet_egoqa")
@dataclass
class Claude35Sonnet_EgoQA(Claude35Sonnet_Base):
    agent_name: str = "QA_Agent"
    max_new_tokens: int = 4096
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    subsampling_factor: int = 2


@register_config("claude35sonnet_egonav")
@dataclass
class Claude35Sonnet_EgoNav(Claude35Sonnet_Base):
    agent_name: str = "EgoNav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 10
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 2


@register_config("claude35sonnet_dminav")
@dataclass
class Claude35Sonnet_DiscreteMapImageNav(Claude35Sonnet_Base):
    agent_name: str = "DiscreteMapImage_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("claude35sonnet_dmtnav")
@dataclass
class Claude35Sonnet_DiscreteMapTextNav(Claude35Sonnet_Base):
    agent_name: str = "DiscreteMapText_Nav_Agent"
    max_new_tokens: int = 2048
    max_history_length: int = 50
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9
    subsampling_factor: int = 1


@register_config("claude35sonnet_cswm_vision")
@dataclass
class Claude35Sonnet_CSWM_Vision(Claude35Sonnet_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "vision"
    max_new_tokens: int = 4096
    max_history_length: int = -1
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9


@register_config("claude35sonnet_cswm_text")
@dataclass
class Claude35Sonnet_CSWM_Text(Claude35Sonnet_Base):
    agent_name: str = "CSWM_Agent"
    task_mode: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = -1
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "low"
    context_truncation_factor: float = 0.9


@register_config("claude35sonnet_mct_vision")
@dataclass
class Claude35Sonnet_MCT_Vision(Claude35Sonnet_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "image"
    max_new_tokens: int = 4096
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    image_detail: str = "high"
    context_truncation_factor: float = 0.9


@register_config("claude35sonnet_mct_text")
@dataclass
class Claude35Sonnet_MCT_Text(Claude35Sonnet_Base):
    agent_name: str = "MCT_Agent"
    description_type: str = "text"
    max_new_tokens: int = 4096
    max_history_length: int = 20
    host_port: Optional[str] = None
    save_dir: Optional[str] = None
    context_truncation_factor: float = 0.9
