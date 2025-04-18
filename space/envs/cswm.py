# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import json
import os.path as osp
import random
from dataclasses import dataclass

from abc import ABC
import cv2
import imageio
import numpy as np

from space.registry import register_env

N_BOXES_TO_MAX_STEPS = {3: 8, 4: 12, 5: 18, 6: 25, 7: 36}


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def get_extents(self):
        return (self.x, self.y, self.x + self.w - 1, self.y + self.h - 1)


class Base_CSWM_Env(ABC):
    r"""Cambridge Spatial Working Memory test:
    N treasures are hidden in N boxes one at a time. Search for the currently hidden treasure by
    selecting a box and opening it. If the treasure is hidden in this box, it is collected and the
    next treasure is hidden in a new box (where the treasure was never hidden before). The goal is
    to collect all the treasures.

    Reference: https://cambridgecognition.com/spatial-working-memory-swm/
    """

    def __init__(self, load_dir: str):
        # Check for text vs. visual version of game
        self.game_mode = (
            "vision" if osp.isfile(osp.join(load_dir, "board.png")) else "text"
        )
        with open(osp.join(load_dir, "states.json"), "r") as fp:
            states = json.load(fp)
        if self.game_mode == "vision":
            self.image = imageio.imread(osp.join(load_dir, "board.png"))
            self.rects = [Rect(x, y, w, h) for x, y, w, h in states["rects"]]
            self.treasure_boxes = [
                Rect(x, y, w, h) for x, y, w, h in states["treasure_boxes"]
            ]
        else:
            self.board_array = np.array(states["board_array"])
            self.rects = states["rects"]
        self.treasures = states["treasures"]
        self.max_steps = N_BOXES_TO_MAX_STEPS[len(self.rects)]
        self.curr_treasure_idx = None
        self.selected_idx = None
        self.n_collected = None
        self.n_steps_taken = None
        self.finished = None
        self.t2r_mapping = None
        self.r2t_mapping = None

    def reset(self):
        self.n_collected = 0
        self.n_steps_taken = 0
        self.curr_treasure_idx = self.treasures[self.n_collected]
        self.selected_idx = None
        self.finished = False
        self.t2r_mapping, self.r2t_mapping = self.sample_random_mapping()
        return self.render(None)

    def sample_random_mapping(self):
        n = len(self.rects)
        mapping = {i: j for i, j in enumerate(np.random.permutation(n).tolist())}
        inv_mapping = {j: i for i, j in mapping.items()}
        return mapping, inv_mapping

    def step(self, action: int):
        assert not self.finished
        assert self.is_valid_action(action)
        action = self.apply_r2t_mapping(action)
        self.selected_idx = action
        done = False
        treasure_collected = None
        if action == self.curr_treasure_idx:
            treasure_collected = self.curr_treasure_idx
            self.n_collected += 1
            if self.n_collected < len(self.rects):
                self.curr_treasure_idx = self.treasures[self.n_collected]
            else:
                done = True
        self.t2r_mapping, self.r2t_mapping = self.sample_random_mapping()
        obs = self.render(treasure_collected)
        self.n_steps_taken += 1
        if self.n_steps_taken >= self.max_steps:
            done = True
        if done:
            self.finished = True
        return obs, 0.0, done, {}

    def render(self, treasure_collected: int):
        raise NotImplementedError

    def apply_r2t_mapping(self, act: int):
        raise NotImplementedError

    def is_valid_action(self, act: int):
        raise NotImplementedError

    def sample_random_action(self):
        raise NotImplementedError


@register_env
class Vision_CSWM_Env(Base_CSWM_Env):
    def render(self, treasure_collected: int):
        font = cv2.FONT_HERSHEY_TRIPLEX
        fontScale = np.ceil(self.image.shape[0] / 600.0).item()
        thickness = int(np.ceil(self.image.shape[0] / 600.0).item())
        image = np.copy(self.image)
        for i, rect in enumerate(self.rects):
            if i == treasure_collected:
                sx = rect.x + 5
                sy = rect.y + 5
                ex = rect.x + rect.w - 6
                ey = rect.y + rect.h - 6
                image = cv2.rectangle(image, (sx, sy), (ex, ey), (255, 191, 0), -1)
            text = f"{self.t2r_mapping[i]}"
            textsize, _ = cv2.getTextSize(text, font, fontScale, thickness)
            text_x = rect.x + (rect.w - textsize[0]) // 2
            text_y = rect.y + rect.h - 1 - (rect.h - textsize[1]) // 2
            cv2.putText(
                image, text, (text_x, text_y), font, fontScale, (0, 0, 0), thickness
            )
        # Fill treasure boxes
        for i in range(self.n_collected):
            r = self.treasure_boxes[i]
            image = cv2.rectangle(
                image, (r.x, r.y), (r.x + r.w - 1, r.y + r.h - 1), (255, 191, 0), -1
            )
        return image

    def apply_r2t_mapping(self, act):
        return self.r2t_mapping[act]

    def is_valid_action(self, act):
        return act >= 0 and act < len(self.rects)

    def sample_random_action(self):
        return random.randint(0, len(self.rects) - 1)


@register_env
class Text_CSWM_Env(Base_CSWM_Env):
    def render(self, treasure_collected: int):
        board_array = np.copy(self.board_array)
        for i, (x, y) in enumerate(self.rects):
            if i == treasure_collected:
                board_array[y, x] = "T"
            else:
                board_array[y, x] = f"{self.t2r_mapping[i] + 1}"
        obs = "Here is the current view of the board. You must find the next treasure. Note that the numbers of the boxes have changed, but the box locations are fixed. Decide which box location you want to open next. Then provide the number associated with the box as the action.\n\n"
        obs += self.convert_array_to_str(board_array) + "\n\n"
        obs += (
            f"Number of treasures collected: {self.n_collected} / {len(self.treasures)}"
        )
        return obs

    def convert_array_to_str(self, array: np.ndarray):
        array_str = []
        for r in array:
            array_str.append(",".join(r.tolist()))
        array_str = "\n".join(array_str)
        return array_str

    def apply_r2t_mapping(self, act):
        return self.r2t_mapping[act - 1]

    def is_valid_action(self, act):
        return act > 0 and act <= len(self.rects)

    def sample_random_action(self):
        return random.randint(1, len(self.rects))
