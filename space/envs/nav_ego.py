# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from space.registry import register_env
from dataclasses import field

import os
import cv2
from typing import Any
from pathlib import Path

try:
    import habitat_sim
    from space.utils.habitat import load_sim, compute_quaternion_from_heading
except ImportError as e:
    print(f"WARNING: Failed to import habitat_sim. Error: {str(e)}")
import numpy as np
import json


GOAL_MAPPING = {
    "CogSci_Env_1_00000": "Painting of a french horn",
    "CogSci_Env_1_00001": "Painting of an aeroplane",
    "CogSci_Env_1_00002": "Painting of a power drill",
    "CogSci_Env_2_00000": "Painting of a turtle",
    "CogSci_Env_2_00001": "Painting of a stove",
    "CogSci_Env_2_00002": "Painting of a cradle",
    "CogSci_Env_3_00000": "Painting of a dog",
    "CogSci_Env_3_00001": "Painting of a camel",
    "CogSci_Env_3_00002": "Painting of a dog",
    "CogSci_Env_4_00000": "Painting of a fish",
    "CogSci_Env_4_00001": "Painting of a meerkat",
    "CogSci_Env_4_00002": "Painting of a guinea pig",
    "CogSci_Env_5_00000": "Painting of a horse cart",
    "CogSci_Env_5_00001": "Painting of a volcano",
    "CogSci_Env_5_00002": "Painting of an aeroplane",
    "CogSci_Env_6_00000": "Painting of a slot machine",
    "CogSci_Env_6_00001": "Painting of a boat",
    "CogSci_Env_6_00002": "Painting of a padlock",
    "CogSci_Env_7_00000": "Painting of a bike",
    "CogSci_Env_7_00001": "Painting of a zebra",
    "CogSci_Env_7_00002": "Painting of a turtle",
    "CogSci_Env_8_00000": "Painting of an ambulance",
    "CogSci_Env_8_00001": "Painting of a soccer ball",
    "CogSci_Env_8_00002": "Painting of a hammer",
    "CogSci_Env_9_00000": "Painting of a hatchet",
    "CogSci_Env_9_00001": "Painting of a bird",
    "CogSci_Env_9_00002": "Painting of a typewriter",
    "CogSci_Env_10_00000": "Painting of a soccer ball",
    "CogSci_Env_10_00001": "Painting of a couch",
    "CogSci_Env_10_00002": "Painting of a fish",
}


@register_env
class NavEgoEnv:
    def __init__(
        self,
        env_dir: str,
        habitat_kwargs: dict[str, Any] = field(default_factory=dict),
        image_downscaling: float = 4.0,
    ):
        """
        Arguments:
            env_dir: Path to directory with environment information
            habitat_kwargs: Keyword args for loading habitat environment
            image_downscaling: Factor to downscale image after rendering
        """

        self._is_sim_initialized = False
        self.scene_name = Path(env_dir).name
        scene_path = os.path.join(env_dir, "scene/scene.glb")
        scene_dataset_config_path = os.path.join(
            env_dir, "scene/scene_dataset_config.json"
        )
        with open(os.path.join(env_dir, "walkthrough_info.json")) as fp:
            info = json.load(fp)
        self.info = info
        self.image_downscaling = image_downscaling
        self.sim = load_sim(scene_path, scene_dataset_config_path, **habitat_kwargs)

    def get_task_info(self):
        start_position = np.array(self.info["walkthrough_info"]["positions"][0])
        start_heading = self.info["walkthrough_info"]["headings"][0]
        goal_position = np.array(self.info["walkthrough_info"]["positions"][-1])
        goal_desc = GOAL_MAPPING[self.scene_name]
        return {
            "start_position": start_position,
            "start_heading": start_heading,
            "goal_position": goal_position,
            "goal_desc": goal_desc,
        }

    def initialize_sim(self, position: np.ndarray, heading_deg: float):
        """
        Arguments:
            position: (x, y, z) array in meters
            heading_deg: heading angle in degrees
        """
        rotation = compute_quaternion_from_heading(heading_deg)
        state = habitat_sim.AgentState(position, rotation)
        self.sim.reset()
        self.sim.initialize_agent(0, state)
        self._is_sim_initialized = True

    def reset(self):
        position = np.array(self.info["walkthrough_info"]["positions"][0])
        heading_deg = self.info["walkthrough_info"]["headings"][0]
        self.initialize_sim(position, heading_deg)
        obs = self.get_observation()
        return obs

    def get_observation(self):
        """
        Get RGB observation from current agent state
        """
        assert self._is_sim_initialized, "Simulator is not initialized."
        obs = self.sim.get_sensor_observations(0)
        return cv2.resize(
            obs["rgb"][..., :3],
            None,
            fx=1.0 / self.image_downscaling,
            fy=1.0 / self.image_downscaling,
        )

    def step(self, act: str):
        assert self._is_sim_initialized, "Simulator is not initialized."
        _ = self.sim.step(act)
        obs = self.get_observation()
        return obs

    def close(self):
        if self.sim is not None:
            self.sim.close()

    def get_sim_state(self):
        assert self._is_sim_initialized
        state = self.sim.agents[0].state
        position = np.array(state.position)
        rotation = state.rotation
        return position, rotation
