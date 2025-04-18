# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import habitat_sim
import numpy as np
import quaternion as qt
from habitat_sim.gfx import LightInfo

EPSILON = 1e-8


def get_rgb_cfg(
    resolution: list[int], agent_height: float
) -> habitat_sim.CameraSensorSpec:
    rgb_cfg = habitat_sim.CameraSensorSpec()
    rgb_cfg.uuid = "rgb"
    rgb_cfg.sensor_type = habitat_sim.SensorType.COLOR
    rgb_cfg.resolution = resolution
    rgb_cfg.position = np.array([0.0, max(agent_height - 0.1, 0.0), 0.0])
    return rgb_cfg


def get_depth_cfg(
    resolution: list[int], agent_height: float
) -> habitat_sim.CameraSensorSpec:
    depth_cfg = habitat_sim.CameraSensorSpec()
    depth_cfg.uuid = "depth"
    depth_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_cfg.resolution = resolution
    depth_cfg.position = np.array([0.0, max(agent_height - 0.1, 0.0), 0.0])
    return depth_cfg


def make_habitat_configuration(
    scene_id: str,
    scene_dataset_config_file: str,
    resolution: list[int],
    agent_height: float = 0.5,
    agent_radius: float = 0.1,
    forward_amount: float = 0.25,
    turn_amount: float = 10.0,
    enable_physics: bool = False,
):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.scene_dataset_config_file = scene_dataset_config_file
    backend_cfg.enable_physics = enable_physics

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        get_rgb_cfg(resolution, agent_height),
        get_depth_cfg(resolution, agent_height),
    ]

    agent_cfg.height = agent_height
    agent_cfg.radius = agent_radius
    agent_cfg.action_space["move_forward"].actuation.amount = forward_amount
    agent_cfg.action_space["turn_left"].actuation.amount = turn_amount
    agent_cfg.action_space["turn_right"].actuation.amount = turn_amount

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

    return sim_cfg


def load_sim(
    scene_id: str,
    scene_dataset_config_file,
    lights: list[LightInfo] = None,
    agent_height: float = 0.5,
    agent_radius: float = 0.1,
    **kwargs,
) -> habitat_sim.Simulator:
    sim_cfg = make_habitat_configuration(
        scene_id,
        scene_dataset_config_file,
        agent_height=agent_height,
        agent_radius=agent_radius,
        **kwargs,
    )
    if lights is not None:
        sim_cfg.sim_cfg.scene_light_setup = "custom_scene_lighting"
        sim_cfg.sim_cfg.override_scene_light_defaults = True
    sim = habitat_sim.Simulator(sim_cfg)

    # Reload navmesh with appropriate height, radius
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = agent_radius
    navmesh_settings.agent_height = agent_height
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    # Add lights if available
    if lights is not None:
        sim.set_light_setup(lights, "custom_scene_lighting")
        sim.reconfigure(sim_cfg)

    return sim


def calculate_geodesic_distance(
    sim: habitat_sim.Simulator, start_position: np.ndarray, goal_position: np.ndarray
) -> tuple[float, bool]:
    """Calculate geodesic distance b/w two points

    Args:
        sim: habitat simulator instance
        start_position: start of path
        goal_position: end of path
    """
    path = habitat_sim.ShortestPath()
    path.requested_start = start_position
    path.requested_end = goal_position
    found_path = sim.pathfinder.find_path(path)
    distance = path.geodesic_distance
    return distance, found_path


def calculate_shortest_path(
    sim: habitat_sim.Simulator, start_position: np.ndarray, goal_position: np.ndarray
) -> tuple[list[np.ndarray], bool]:
    """Calculate shortest path b/w two points

    Args:
        sim: habitat simulator instance
        start_position: start of path
        goal_position: end of path
    """
    path = habitat_sim.ShortestPath()
    path.requested_start = start_position
    path.requested_end = goal_position
    found_path = sim.pathfinder.find_path(path)
    return path.points, found_path


def quaternion_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> qt.quaternion:
    r"""Computes the quaternion representation of v1 using v0 as the origin."""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    c = v0.dot(v1)
    # Epsilon prevents issues at poles.
    if c < (-1 + EPSILON):
        c = max(c, -1)
        m = np.stack([v0, v1], 0)
        _, _, vh = np.linalg.svd(m, full_matrices=True)
        axis = vh.T[:, 2]
        w2 = (1 + c) * 0.5
        w = np.sqrt(w2)
        axis = axis * np.sqrt(1 - w2)
        return qt.quaternion(w, *axis)

    axis = np.cross(v0, v1)
    s = np.sqrt((1 + c) * 2)
    return qt.quaternion(s * 0.5, *(axis / s))


def quaternion_rotate_vector(quat: qt.quaternion, v: np.ndarray) -> np.ndarray:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.ndarray: The rotated vector
    """
    vq = qt.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def compute_heading_from_quaternion(r) -> float:
    """
    r - rotation quaternion

    Computes clockwise rotation about Y.
    """
    # quaternion - np.quaternion unit quaternion
    # Real world rotation
    direction_vector = np.array([0, 0, -1])  # Forward vector
    heading_vector = quaternion_rotate_vector(r.inverse(), direction_vector)

    phi = -np.arctan2(heading_vector[0], -heading_vector[2]).item()
    return phi


def compute_quaternion_from_heading(h_deg: float) -> qt.quaternion:
    """Calculates quaternion corresponding to heading.

    Args:
        h_deg: Clockwise rotation about Y in degrees.
    """
    h = np.deg2rad(h_deg)
    fwd_dir = np.array([0.0, 0.0, -1.0])
    head_dir = np.array([np.sin(h), 0.0, -np.cos(h)])
    quat = quaternion_from_two_vectors(fwd_dir, head_dir)
    return quat


def quaternion_to_list(q: qt.quaternion):
    return q.imag.tolist() + [q.real]


class DistanceToGoal:
    def __init__(self, sim: habitat_sim.Simulator, goal_position: np.ndarray):
        """
        Class to compute distance to goal metric

        Arguments:
            sim: habitat simulator instance
            goal_position: (x, y, z) location of goal (in meters)
        """
        self.sim = sim
        self.goal_position = goal_position

    def __call__(self, trajectory_positions: list[np.ndarray]) -> float:
        last_position = trajectory_positions[-1]
        distance, found_path = calculate_geodesic_distance(
            self.sim, last_position, self.goal_position
        )
        assert found_path, "Could not find a path in DistanceToGoal"
        return distance


class Success:
    def __init__(
        self,
        sim: habitat_sim.Simulator,
        goal_position: np.ndarray,
        dist_thresh: float = 1.0,
    ):
        """
        Class to compute success metric

        Arguments:
            sim: habitat simulator instance
            goal_position: (x, y, z) location of goal viewpoint (in meters)
            dist_thresh: geodesic distance threshold to determine success (in meters)
        """
        self.sim = sim
        self.goal_position = goal_position
        self.dist_thresh = dist_thresh
        self._d2g = DistanceToGoal(sim, goal_position)

    def __call__(
        self,
        last_action_was_stop: bool,
        trajectory_positions: list[np.ndarray],
    ) -> float:
        # If last action called was not STOP, then success is 0 by definition
        if not last_action_was_stop:
            success = 0.0
        else:
            d2g = self._d2g(trajectory_positions)
            if d2g <= self.dist_thresh:
                success = 1.0
            else:
                success = 0.0
        return success


class SPL:
    def __init__(
        self,
        sim: habitat_sim.Simulator,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        dist_thresh: float = 1.0,
    ):
        """
        Class to compute Success weighted by Path Length

        Reference: https://arxiv.org/pdf/1807.06757.pdf

        Arguments:
            sim: habitat simulator instance
            start_position: (x, y, z) location of start (in meters)
            goal_position: (x, y, z) location of goal viewpoint (in meters)
            dist_thresh: geodesic distance threshold to determine success (in meters)
        """
        self.sim = sim
        self._success = Success(sim, goal_position, dist_thresh)
        # Calculate shortest path length
        distance, found_path = calculate_geodesic_distance(
            sim, start_position, goal_position
        )
        assert found_path, "Could not find a path in SPL.__init__()"
        self._shortest_path_length = distance

    def __call__(
        self, last_action_was_stop: bool, trajectory_positions: list[np.ndarray]
    ):
        success = self._success(last_action_was_stop, trajectory_positions)
        if success == 0.0:
            spl = 0.0
        else:
            current_path_length = self.calculate_path_length(trajectory_positions)
            spl = max(self._shortest_path_length, EPSILON) / max(
                current_path_length, self._shortest_path_length, EPSILON
            )
        return spl

    def calculate_path_length(self, trajectory_positions: list[np.ndarray]):
        path_length = 0.0
        for p1, p2 in zip(trajectory_positions[:-1], trajectory_positions[1:]):
            distance, found_path = calculate_geodesic_distance(self.sim, p1, p2)
            assert found_path, "Could not find a path in SPL.calculate_path_length()"
            path_length += distance
        return path_length
