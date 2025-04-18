# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
AGENTS_REGISTRY = {}
CONFIGS_REGISTRY = {}
ENVS_REGISTRY = {}


def register_agent(cls):
    global AGENTS_REGISTRY
    AGENTS_REGISTRY[cls.__name__] = cls
    return cls


def register_config(name: str):
    def decorator(func):
        global CONFIGS_REGISTRY
        assert name not in CONFIGS_REGISTRY
        CONFIGS_REGISTRY[name] = func
        return func

    return decorator


def register_env(cls: str):
    global ENVS_REGISTRY
    ENVS_REGISTRY[cls.__name__] = cls
    return cls
