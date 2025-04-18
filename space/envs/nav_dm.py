# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from space.registry import register_env

import copy
import json
import random

import cv2
import networkx as nx
import numpy as np

from space.utils.visualizations import add_text_to_image


def get_angle_between_vectors(u: np.ndarray, v: np.ndarray):
    # Measure absolute angle between two vectors (in radians)
    # Reference: https://github.com/pytorch/pytorch/issues/59194
    assert len(u) == 2
    assert len(v) == 2

    un = u / np.linalg.norm(u)
    vn = v / np.linalg.norm(v)

    y = un - vn
    x = un + vn

    a0 = 2 * np.arctan(np.linalg.norm(y) / np.linalg.norm(x))

    # Measure sign of angle using cross-product
    # Reference: https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
    u_ext = np.concatenate([un, np.array([0.0])], axis=0)
    v_ext = np.concatenate([vn, np.array([0.0])], axis=0)
    dot_cross = np.dot(np.cross(u_ext, v_ext), np.array([0.0, 0.0, 1.0]))
    sign = np.sign(dot_cross).item()

    if (not np.signbit(a0)) or np.signbit(np.pi - a0):
        return a0 * sign
    elif np.signbit(a0):
        return 0.0 * sign
    else:
        return np.pi * sign


@dataclass
class Position:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)


def get_distance(pA: Position, pB: Position) -> float:
    return np.sqrt((pA.x - pB.x) ** 2 + (pA.y - pB.y) ** 2).item()


def compute_success(
    gt_path: list[Position], pred_path: list[Position], threshold: float = 0.0
) -> float:
    d2g = compute_distance2goal(gt_path, pred_path)
    return float(d2g <= threshold)


def compute_distance2goal(gt_path: list[Position], pred_path: list[Position]) -> float:
    gt_pos = gt_path[-1]
    pred_pos = pred_path[-1]
    return get_distance(gt_pos, pred_pos)


def compute_path_length(positions: list[Position]):
    pl = 0.0
    for p1, p2 in zip(positions[:-1], positions[1:]):
        pl += get_distance(p1, p2)
    return pl


def compute_spl(
    gt_path: list[Position],
    pred_path: list[Position],
    threshold: float = 0.0,
    eps: float = 1e-10,
) -> float:
    succ = compute_success(gt_path, pred_path, threshold=threshold)
    shortest_pl = compute_path_length(gt_path)
    pred_pl = compute_path_length(pred_path)
    spl = succ * shortest_pl / max([shortest_pl, pred_pl, eps])
    return spl


def calculate_path_validity(maze: np.ndarray, pred_path: list[Position]):
    valid_path = True
    max_idx = len(pred_path) - 1
    # Ensure all locations on path are navigable
    for idx, pos in enumerate(pred_path):
        if pos.x < 0 or pos.x >= maze.shape[1] or pos.y < 0 or pos.y >= maze.shape[0]:
            max_idx = idx - 1
            break
        if maze[pos.y, pos.x] == 0:
            valid_path = False
            max_idx = idx - 1
            break
    # Ensure consecutive positions are only 1 step apart
    for idx, (pA, pB) in enumerate(zip(pred_path[:-1], pred_path[1:])):
        if not get_distance(pA, pB) <= 1.0:
            valid_path = False
            max_idx = min(max_idx, idx)
            break

    if max_idx < 0:
        valid_pred_subset = []
    else:
        valid_pred_subset = pred_path[: max_idx + 1]
    return float(valid_path), valid_pred_subset


def evaluate_path_efficiency(
    gt_path: list[tuple[int, int]],
    pred_path: list[tuple[int, int]],
    maze: np.ndarray,
    issued_stop: bool = True,
    dist_thresh: float = 0.0,
) -> dict[str, float]:
    gt_path = [Position(c, r) for (r, c) in gt_path]
    pred_path = [Position(c, r) for (r, c) in pred_path]
    # Check if path is valid or not
    if len(pred_path) > 0:
        valid_path, valid_pred_subset = calculate_path_validity(maze, pred_path)
    else:
        valid_path, valid_pred_subset = False, []

    if len(valid_pred_subset) > 0:
        # Sanity check
        assert calculate_path_validity(maze, valid_pred_subset)[0] == 1
        if issued_stop:
            metrics = {
                "valid_path": valid_path,
                "success": compute_success(gt_path, valid_pred_subset, dist_thresh),
                "spl": compute_spl(gt_path, valid_pred_subset, dist_thresh),
            }
        else:
            metrics = {
                "valid_path": valid_path,
                "success": 0.0,
                "spl": 0.0,
            }
    else:
        metrics = {
            "valid_path": 0.0,
            "success": 0.0,
            "spl": 0.0,
            "softspl": 0.0,
        }

    return metrics


@register_env
class NavDiscreteMapEnv:
    valid_observation_types = ["text", "image"]
    num_pixels_per_cell = 100

    def __init__(
        self,
        env_path: str,
        local_context: int,
        observation_type: str,
    ):
        assert local_context % 2 == 1
        assert observation_type in self.valid_observation_types
        self.local_context = local_context
        self.observation_type = observation_type

        with open(env_path, "r") as fp:
            data = json.load(fp)
        self.textmap = np.array(
            data["map"]
        )  # 0 = obstacles, 1/A/B/C/... = free space, landmarks = A/B/C/...
        self.landmarks = data["landmarks"]
        self.start_location = data["start_location"]
        self.goal_location = data["goal_location"]
        self.walkthrough_waypoint_locations = data["walkthrough_waypoint_locations"]

        self.goal_name = None
        self.collided = None
        for lname, lpos in self.landmarks.items():
            if self.goal_location == lpos:
                self.goal_name = lname
                break
        assert self.goal_name is not None
        self.current_location = None

        graph = nx.Graph()
        H, W = self.textmap.shape
        # Add nodes
        for x in range(W):
            for y in range(H):
                if self.textmap[y, x] == "0":
                    continue
                graph.add_node((x, y))
        # Add edges
        for x in range(W):
            for y in range(H):
                if self.textmap[y, x] == "0":
                    continue
                # Only add edges to forward-facing neighbors
                nbs = [(x, y + 1), (x + 1, y)]
                for x_, y_ in nbs:
                    if x_ >= W or y_ >= H:
                        continue
                    if self.textmap[y_, x_] == "0":
                        continue
                    graph.add_edge((x, y), (x_, y_))

        self.graph = graph

    def reset(self, walkthrough_mode: bool = False):
        self.current_location = self.start_location
        self.collided = False
        obs = self.get_observation(walkthrough_mode)
        return obs

    def step(self, action: str, walkthrough_mode: bool = False):
        self.collided = self.update_position(action)
        obs = self.get_observation(walkthrough_mode)
        return obs

    def update_position(self, action: str):
        assert action in ["up", "down", "left", "right"]
        x, y = self.current_location
        if action == "up":
            x_, y_ = x, y - 1
        elif action == "down":
            x_, y_ = x, y + 1
        elif action == "left":
            x_, y_ = x - 1, y
        elif action == "right":
            x_, y_ = x + 1, y
        if (
            x_ < 0
            or x_ >= self.textmap.shape[1]
            or y_ < 0
            or y_ >= self.textmap.shape[0]
        ):
            collided = True
        elif self.textmap[y_, x_] == "0":
            collided = True
        else:
            collided = False
            x, y = x_, y_
        self.current_location = [x, y]
        return collided

    def get_observation(self, walkthrough_mode: bool = False):
        localmap = self.get_local_context_array()
        if self.observation_type == "text":
            obs = self.render_text(localmap, walkthrough_mode)
        elif self.observation_type == "image":
            obs = self.render_vision(localmap, walkthrough_mode)
        return obs

    def get_local_context_array(self):
        # Pad text map to avoid out-of-bound issues
        pad_size = (self.local_context - 1) // 2
        textmap = np.pad(self.textmap, pad_size, mode="constant", constant_values="0")
        x, y = self.current_location
        x += pad_size
        y += pad_size
        localmap = textmap[
            (y - pad_size) : (y + pad_size + 1), (x - pad_size) : (x + pad_size + 1)
        ]
        return localmap

    def render_text(self, localmap: np.ndarray, walkthrough_mode: bool = False):
        obs = []
        if not walkthrough_mode:
            if self.collided:
                obs.append("Oops. You previous action resulted in a collision.")
            obs.append(
                'Here is a birds-eye view of the {}x{} area surrounding your current position. You are located at the center of this view. Your position is denoted by "*".\n\n'.format(
                    self.local_context, self.local_context
                )
            )
        localmap = np.copy(localmap)
        landmarks_visible = []
        for x in range(localmap.shape[1]):
            for y in range(localmap.shape[0]):
                if localmap[y, x] != "0" and localmap[y, x] != "1":
                    landmarks_visible.append(localmap[y, x].item())
        H, W = localmap.shape
        localmap[H // 2, W // 2] = "*"
        localmap_str = []
        for row in localmap:
            localmap_str.append(",".join(row.tolist()))
        localmap_str = "\n".join(localmap_str)
        if not walkthrough_mode:
            localmap_str = "```\n" + localmap_str + "\n```"
        localmap_str += "\n"
        obs.append(localmap_str)
        # Add information about landmarks visible
        if len(landmarks_visible) > 0 and not walkthrough_mode:
            obs.append(
                "The landmarks visible in your local context are: {}. Note that the landmark locations are also navigable spaces, i.e., you can move over them.".format(
                    ",".join(landmarks_visible)
                )
            )
        # Add information about goal
        if not walkthrough_mode:
            if self.current_location != self.goal_location:
                obs.append(f"Your objective is to reach landmark: {self.goal_name}")
            else:
                obs.append(
                    f"Congrats! You've reached the goal landmark {self.goal_name}."
                )
        obs = "\n".join(obs)
        return obs

    def render_vision(self, localmap: np.ndarray, walkthrough_mode: bool = False):
        N = self.num_pixels_per_cell
        H, W = localmap.shape
        image = np.zeros((N * H, N * W, 3), dtype=np.uint8)
        ############################################################################################
        # Apply Pacman-style coloring
        # ---------------------------
        # Blue - walls
        # Black - free spaces
        # Yellow - current position
        # Blue with text internally - landmarks
        ############################################################################################
        # Color free-spaces
        for r in range(H):
            for c in range(W):
                if localmap[r, c] != "0":
                    image[N * r : N * (r + 1), N * c : N * (c + 1), :] = np.array(
                        [0, 0, 255]
                    )
        # Draw grid lines
        for r in range(H):
            for c in range(W):
                image = cv2.rectangle(
                    image,
                    (c * N, r * N),
                    ((c + 1) * N, (r + 1) * N),
                    (255, 255, 255),
                    2,
                )

        # Color current position
        r, c = H // 2, W // 2
        start_x = int(c * N + N * 0.2)
        start_y = int(r * N + N * 0.2)
        end_x = int(c * N + N * 0.8)
        end_y = int(r * N + N * 0.8)
        image = cv2.rectangle(
            image, (start_x, start_y), (end_x, end_y), (255, 255, 0), -1
        )
        # Draw landmarks
        mrows, mcols = np.where((localmap != "0") & (localmap != "1"))
        if len(mrows) > 0:
            for r, c in zip(mrows, mcols):
                center_x = c * N + N // 2
                center_y = r * N + N // 2
                image = cv2.circle(
                    image, (center_x, center_y), int(N * 0.25), (255, 0, 0), -1
                )
                image = add_text_to_image(
                    image,
                    localmap[r, c].item(),
                    origin=(center_x - 10, center_y + 10),
                    color=(255, 255, 255),
                )
        ############################################################################################
        # Add status message below image: Did the agent collide after the previous action?
        ############################################################################################
        if not walkthrough_mode:
            status_size = int(0.1 * N * W)
            status_image = np.full((status_size, N * W, 3), 128, dtype=np.uint8)
            status_image[:5, :, :] = np.array([255, 255, 255])
            if self.collided:
                text = "The previous action caused you to collide into a wall."
                font = cv2.FONT_HERSHEY_TRIPLEX
                fontScale = np.ceil(status_size / 200.0).item()
                thickness = int(np.ceil(status_size / 200.0).item())
                textsize, _ = cv2.getTextSize(text, font, fontScale, thickness)
                textX = (status_image.shape[1] - textsize[0]) // 2
                textY = (status_image.shape[0] + textsize[1]) // 2
                status_image = cv2.putText(
                    status_image,
                    text,
                    (textX, textY),
                    font,
                    fontScale,
                    (255, 255, 255),
                    thickness,
                )
            image = np.concatenate([image, status_image], axis=0)

        return image

    def generate_walkthrough_video(self):
        def _get_action_from_positions(pos_1, pos_2):
            x1, y1 = pos_1
            x2, y2 = pos_2
            assert not ((x1 != x2) and (y1 != y2))
            if x1 > x2:
                act = "left"
            elif x1 < x2:
                act = "right"
            elif y1 > y2:
                act = "up"
            elif y1 < y2:
                act = "down"
            else:
                raise ValueError("Should not reach here!")
            return act

        ############################################################################################
        # Generate walkthrough video
        ############################################################################################
        current_position = self.start_location
        walkthrough_obs = []
        walkthrough_info = {"positions": [], "actions": []}
        walkthrough_info["positions"].append(current_position)
        for waypoint_position in self.walkthrough_waypoint_locations + [
            self.goal_location
        ]:
            path = nx.shortest_path(
                self.graph, tuple(current_position), tuple(waypoint_position)
            )
            walkthrough_info["positions"] += [[x, y] for x, y in path[1:]]
            current_position = waypoint_position
        for p1, p2 in zip(
            walkthrough_info["positions"][:-1], walkthrough_info["positions"][1:]
        ):
            walkthrough_info["actions"].append(_get_action_from_positions(p1, p2))
        obs = self.reset(walkthrough_mode=True)
        walkthrough_obs.append(obs)
        for act in walkthrough_info["actions"]:
            obs = self.step(act, walkthrough_mode=True)
            walkthrough_obs.append(obs)
        assert self.goal_location == self.current_location
        ############################################################################################
        # Generate shortest-path video
        ############################################################################################
        shortestpath_obs = []
        shortestpath_info = {"positions": [], "actions": []}
        shortestpath_positions = nx.shortest_path(
            self.graph, tuple(self.start_location), tuple(self.goal_location)
        )
        shortestpath_info["positions"] = [[x, y] for x, y in shortestpath_positions]
        for p1, p2 in zip(
            shortestpath_info["positions"][:-1], shortestpath_info["positions"][1:]
        ):
            shortestpath_info["actions"].append(_get_action_from_positions(p1, p2))
        obs = self.reset(walkthrough_mode=True)
        shortestpath_obs.append(obs)
        for act in shortestpath_info["actions"]:
            obs = self.step(act, walkthrough_mode=True)
            shortestpath_obs.append(obs)
        assert self.goal_location == self.current_location

        return walkthrough_obs, walkthrough_info, shortestpath_obs, shortestpath_info

    def place_next_to_landmark(self, landmark_name: str):
        H, W = self.textmap.shape
        x, y = self.landmarks[landmark_name]
        assert self.textmap[y, x].item() == landmark_name
        # Search neighborhood for free-spaces next to landmark
        nbs = [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]
        free_nbs = []
        for x_, y_ in nbs:
            if x_ < 0 or x_ >= W or y_ < 0 or y_ >= H:
                continue
            if self.textmap[y_, x_] == "1":
                free_nbs.append([x_, y_])
        assert len(free_nbs) > 0
        # Sample a random location to stand next to landmark
        x_n, y_n = random.choice(free_nbs)
        self.current_location = [x_n, y_n]

    def get_angle_to_landmarks(self, landmark_name_1: str, landmark_name_2: str):
        x1, y1 = self.landmarks[landmark_name_1]
        x2, y2 = self.landmarks[landmark_name_2]
        xc, yc = self.current_location
        assert not (xc == x1 and yc == y1)
        assert not (xc == x2 and yc == y2)
        u = np.array([x1 - xc, y1 - yc])
        v = np.array([x2 - xc, y2 - yc])
        angle = get_angle_between_vectors(u, v)
        angle = np.rad2deg(angle).item()
        return angle

    def get_distance_between_landmarks(
        self, landmark_name_1: str, landmark_name_2: str
    ):
        x1, y1 = self.landmarks[landmark_name_1]
        x2, y2 = self.landmarks[landmark_name_2]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2).item()

    def get_map_sketch(self):
        landmarks = copy.deepcopy(self.landmarks)
        landmarks["start"] = self.start_location
        return landmarks
