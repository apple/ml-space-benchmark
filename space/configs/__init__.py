# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from space.registry import CONFIGS_REGISTRY
from dataclasses import asdict

import space.configs.claude  # noqa
import space.configs.gpt  # noqa
import space.configs.llama  # noqa
import space.configs.mistral  # noqa
import space.configs.phi  # noqa
import space.configs.yi  # noqa


def get_config(name: str):
    assert name in CONFIGS_REGISTRY
    cfg = CONFIGS_REGISTRY[name]()
    return asdict(cfg)
