# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any
from space.registry import AGENTS_REGISTRY
import space.agents.cswm_agent  # noqa
import space.agents.dmnav_agent  # noqa
import space.agents.egonav_agent  # noqa
import space.agents.mct_agent  # noqa
import space.agents.qa_agent  # noqa


def get_agent(agent_name: str, agent_cfg: dict[str, Any]):
    assert agent_name in AGENTS_REGISTRY
    agent_cls = AGENTS_REGISTRY[agent_name]
    agent = agent_cls(**agent_cfg)
    return agent
