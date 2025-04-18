# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from space.registry import ENVS_REGISTRY

import space.envs.cswm  # noqa
import space.envs.mct  # noqa
import space.envs.nav_dm  # noqa
import space.envs.nav_ego  # noqa


def get_env(name: str, *args, **kwargs):
    assert name in ENVS_REGISTRY
    env = ENVS_REGISTRY[name](*args, **kwargs)
    return env
