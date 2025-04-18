# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import json
import os

import cv2
import networkx as nx
import numpy as np

from space.registry import register_env
from space.envs.nav_dm import evaluate_path_efficiency as evaluate_path_efficiency


@register_env
class MCT_Env:
    valid_actions: list[str] = ["up", "down", "left", "right"]
    valid_description_types: list[str] = ["image", "text"]

    def __init__(
        self,
        env_dir: str,
        description_type: str,
        num_pixels_per_cell: int = 100,
    ):
        assert description_type in self.valid_description_types
        self.description_type = description_type
        self.num_pixels_per_cell = num_pixels_per_cell

        with open(os.path.join(env_dir, "info.json")) as fp:
            info = json.load(fp)
        self.start = tuple(info["start"])  # (r, c)
        self.goal = tuple(info["goal"])  # (r, c)
        self.maze = np.load(os.path.join(env_dir, "maze.npz"))["maze"]

        # Create graph for shortest-path planning
        self.create_graph_from_maze()
        self.shortest_path = self.get_shortest_path_from_nodes(self.start, self.goal)

        # Navigation state maintenance
        self.current = None
        self.steps_taken = None
        self.actions_taken = None
        self.collided = None
        self.path_taken = None

    def reset(self):
        self.current = self.start
        self.steps_taken = 0
        self.actions_taken = []
        self.collided = False
        self.path_taken = [self.current]
        obs = self.get_observation()
        return obs, {"has_collided": self.collided}

    def step(self, action: str):
        self.collided = self._update_step(action)
        self.actions_taken.append(action)
        obs = self.get_observation()
        return obs, {"has_collided": self.collided}

    def _update_step(self, action: str):
        assert action in self.valid_actions
        r, c = self.current
        if action == "left":
            next_pos = (r, c - 1)
        elif action == "right":
            next_pos = (r, c + 1)
        elif action == "up":
            next_pos = (r - 1, c)
        else:
            next_pos = (r + 1, c)
        if next_pos in self.nodes:
            self.current = next_pos
            has_collided = False
        else:
            has_collided = True
        self.steps_taken += 1
        return has_collided

    def create_graph_from_maze(self):
        H, W = self.maze.shape[:2]
        self.graph = nx.Graph()
        self.nodes = set()
        ## Add nodes
        for r in range(H):
            for c in range(W):
                if self.maze[r, c] == 1:
                    self.graph.add_node((r, c))
                    self.nodes.add((r, c))
        ## Add edges
        for r in range(H):
            for c in range(W):
                if self.maze[r, c] == 1:
                    # Check neighbors (only forward looking)
                    nbs = [(r + 1, c), (r, c + 1)]
                    for r_, c_ in nbs:
                        if not (r_ >= 0 and r_ < H and c_ >= 0 and c_ < W):
                            continue
                        if self.maze[r_, c_] == 1:
                            self.graph.add_edge((r, c), (r_, c_))

    def get_shortest_path_from_nodes(
        self, start_node: tuple[int, int], goal_node: tuple[int, int]
    ):
        assert start_node in self.nodes
        assert goal_node in self.nodes
        return nx.shortest_path(self.graph, start_node, goal_node)

    def get_observation(self):
        if self.description_type == "image":
            return self.render_visual_observation(self.maze, self.current, self.goal)
        elif self.description_type == "text":
            return self.render_textual_observation(self.maze, self.current, self.goal)
        else:
            raise ValueError(
                f"get_observation() is not defined for description type: {self.description_type}"
            )

    def render_visual_observation(
        self, maze: np.ndarray, current_loc: tuple[int, int], goal_loc: tuple[int, int]
    ):
        N = self.num_pixels_per_cell
        H, W = maze.shape
        image = np.zeros((N * H, N * W, 3), dtype=np.uint8)
        ############################################################################################
        # Apply Pacman-style coloring
        # ---------------------------
        # Blue - walls
        # Black - free spaces
        # Yellow - current position
        # Red - goal position
        ############################################################################################
        # Color walls
        for r in range(H):
            for c in range(W):
                if maze[r, c] == 0:
                    image[N * r : N * (r + 1), N * c : N * (c + 1), :] = np.array(
                        [0, 0, 255]
                    )
        # Color current position
        r, c = current_loc
        start_x = int(c * N + N * 0.2)
        start_y = int(r * N + N * 0.2)
        end_x = int(c * N + N * 0.8)
        end_y = int(r * N + N * 0.8)
        image = cv2.rectangle(
            image, (start_x, start_y), (end_x, end_y), (255, 255, 0), -1
        )
        # Color goal position (if visible)
        r, c = goal_loc
        if r >= 0 and r < H and c >= 0 and c < W:
            center_x = c * N + N // 2
            center_y = r * N + N // 2
            image = cv2.circle(
                image, (center_x, center_y), int(N * 0.25), (255, 0, 0), -1
            )
        ############################################################################################
        # Add status message below image: Did the agent collide after the previous action?
        ############################################################################################
        status_size = int(0.1 * N * W)
        status_image = np.full((status_size, N * W, 3), 128, dtype=np.uint8)
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

    def render_textual_observation(
        self,
        maze: np.ndarray,
        current_loc: tuple[int, int],
        goal_loc: tuple[int, int],
        add_positional_information: bool = True,
    ):
        desc = ""
        if self.collided:
            desc += "Collision alert: The previous action caused you to collide into a wall.\n\n"
        desc += "Here is the current view of the maze.\n\n"
        # Array-like description of maze
        maze_str = np.array([[str(int(col)) for col in row] for row in maze])
        maze_str[current_loc[0], current_loc[1]] = "A"
        maze_str[goal_loc[0], goal_loc[1]] = "G"
        for row in maze_str:
            desc += ",".join(row.tolist()) + "\n"
        desc += "\n\n"
        desc += "0 represents obstacles. 1 represents free spaces. G is the goal. A is your current position in the maze.\n"
        if add_positional_information:
            desc += (
                "Your current location in the maze is row, column = ({}, {}).\n".format(
                    *current_loc
                )
            )
            desc += "The goal location is row, column = ({}, {}).\n".format(*goal_loc)

        return desc
