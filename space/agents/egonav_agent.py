# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import re

from typing import Any
import numpy as np

from space.utils.common import get_image_as_message

from space.registry import register_agent
from space.agents.basenav_agent import Base_Navigation_Agent


TASK_PROMPT_ROUTE_FOLLOWING = """You are a sentient living creature capable of navigating in environments, building internal spatial representations of environments, and finding goals in them. You will be shown a video of the shortest route from the initial position to the goal. You must look at the video and understand the environment structure and the route taken. Then, you will be placed in the environment at the same initial position. You must navigate from the initial position to the goal using the same route shown in the video, as quickly as possible. Below, you will find sections highlighting more details about the task. You can refer to these for more information.

OBSERVATIONS:
The images are recorded from a perspective viewpoint (i.e., egocentric or first-person). This means that you are likely to see objects from different angles, resulting in a skewed appearance of the underlying 3D objects. It is important for you to look past this skew in the appearance and percive the true shape of the object in 3D.

GOAL:
You will be provided an object goal using a text description and an image of the object. You must find the goal object in the environment by repeating the path shown in the video walkthrough. Once you find it, move close to the location of the goal and re-orient yourself to face the object.

ACTIONS:
You have four actions available.

move_forward: move forward by 0.25m along the current heading direction. It does not change the heading angle.
turn_left: decrease your heading angle by 30 degrees. It does not change the (x, y) position.
turn_right: increase your heading angle by 30 degrees. It does not change the (x, y) position.
stop: ends the current task. Issue this action only if you think you have reached the goal. If you haven't reached the goal, this action will result in a navigation failure that cannot be recovered from.

STUCK IN PLACE BEHAVIOR:
Avoid getting stuck in one place, i.e., do not alternate between left and right turns without going anywhere. You must try and move around consistently without being stuck in one place.

STOP CRITERIA:
Before executing stop, you must ensure that you've "reached" the goal correctly. To reach a goal, you have to move close enough to the wall where you see the goal, and see the object clearly in your observation in front of you.

RESPONSE FORMAT:
Respond in the following format:

Reasoning: <text explanation string in one or two short sentences --- provide all your explanations and inner thoughts here - avoid verbosity and be concise>
Intent: <state your intent in one short sentence, i.e., what you are trying to achieve>
Then provide the final action to take in a json formatted string.
```
{
    "action": <action name -- must be one of move_forward, turn_left, turn_right, stop>
}
```
"""

TASK_PROMPT_NOVEL_SHORTCUTS = """You are a sentient living creature capable of navigating in environments, building internal spatial representations of environments, and finding goals in them. You will be shown a video of some route from the initial position to the goal. You must look at the video and understand the environment structure, and remember the locations of the start and the goal. The video may show a long-winded route from the start to the goal with unnecessary detours. Based on the environment structure, you must identify a faster route to the goal. Then, you will be placed in the environment at the same initial position. You must navigate to the goal using your identified shortest route as quickly as possible. Below, you will find sections highlighting more details about the task. You can refer to these for more information.

OBSERVATIONS:
The images are recorded from a perspective viewpoint (i.e., egocentric or first-person). This means that you are likely to see objects from different angles, resulting in a skewed appearance of the underlying 3D objects. It is important for you to look past this skew in the appearance and percive the true shape of the object in 3D.

GOAL:
You will be provided an object goal using a text description and an image of the object. You must find the goal object in the environment by identifying the shortest route based on your experience from the video. Once you find the goal, move close to its location and re-orient yourself to face the object.

ACTIONS:
You have four actions available.

move_forward: move forward by 0.25m along the current heading direction. It does not change the heading angle.
turn_left: decrease your heading angle by 30 degrees. It does not change the (x, y) position.
turn_right: increase your heading angle by 30 degrees. It does not change the (x, y) position.
stop: ends the current task. Issue this action only if you think you have reached the goal. If you haven't reached the goal, this action will result in a navigation failure that cannot be recovered from.

STUCK IN PLACE BEHAVIOR:
Avoid getting stuck in one place, i.e., do not alternate between left and right turns without going anywhere. You must try and move around consistently without being stuck in one place.

STOP CRITERIA:
Before executing stop, you must ensure that you've "reached" the goal correctly. To reach a goal, you have to move the robot close enough to the wall where you see the goal, and see the object clearly in your observation in front of you.

RESPONSE FORMAT:
Respond in the following format:

Reasoning: <text explanation string in one or two short sentences --- provide all your explanations and inner thoughts here - avoid verbosity and be concise>
Intent: <state your intent in one short sentence, i.e., what you are trying to achieve>
Then provide the final action to take in a json formatted string.
```
{
    "action": <action name -- must be one of move_forward, turn_left, turn_right, stop>
}
```
"""


@register_agent
class EgoNav_Agent(Base_Navigation_Agent):
    def set_task_prompt(self, walkthrough_key: str):
        assert walkthrough_key in ["shortestpath", "walkthrough"]
        if walkthrough_key == "shortestpath":
            # Route following experiment
            self.task_prompt = TASK_PROMPT_ROUTE_FOLLOWING
        else:
            # Novel shortcuts experiment
            self.task_prompt = TASK_PROMPT_NOVEL_SHORTCUTS

    def get_goal_prompt(self, goal_desc: str, goal_img: np.ndarray):
        message = [
            f"Now, you must navigate to the goal. Here is the goal description and the image: {goal_desc}",
            get_image_as_message(
                image=goal_img,
                model_name=self.model_name,
                image_detail=self.image_detail,
            ),
        ]
        return message

    def get_action(self, obs: np.ndarray):
        obs_encoded = get_image_as_message(
            image=obs, model_name=self.model_name, image_detail=self.image_detail
        )
        new_message = self.get_observation_prompt(obs_encoded) + [
            "What action do you select next? The available actions are move_forward, turn_left, turn_right and stop. Recall that each turn action is only 30 degrees and each forward step is only 0.25m, so you may have to execute several actions to notice substantial changes in your viewpoints. Be patient and persist with your actions over a longer time horizon.",
        ]
        response_txt = self.handle_model_response(new_message)
        ############################################################################################
        # Clean-up context history
        # Remove explanatory text content of last user message "What actions do you select next? ...."
        assert self.dialog.history[-2]["role"] == "user"
        self.dialog.history[-2] = {
            "role": "user",
            "content": self.get_clean_observation_prompt(obs_encoded),
        }
        ############################################################################################
        action = self.convert_response_to_action(response_txt)
        if action is None:
            # Turn left if no action was parseable from the model
            action = "turn_left"

        return action

    def get_observation_prompt(self, image_encoded: dict[str, Any]) -> list[str]:
        return [
            "Here is the current observation. If you are stuck very close to the same wall for several steps, it means that you are colliding and need to turn around and search elsewhere.",
            image_encoded,
        ]

    def get_clean_observation_prompt(self, image_encoded: str):
        return ["Here is the current observation.", image_encoded]

    def get_walkthrough_prompt(self, walkthrough_frames: np.ndarray) -> list[str]:
        if self.walkthrough_key == "shortestpath":
            output = [
                "Here are the sequence of frames from the walkthrough video demonstrating the route you need to take. Analyze the walkthrough to understand the movements and the maze structure. Take a note of all the details needed to help you repeat this route when navigating next. Think step by step."
            ]
        elif self.walkthrough_key == "walkthrough":
            output = [
                "Here are the sequence of frames from the walkthrough video demonstrating a suboptimal route from the start to some goal location. Analyze the walkthrough to understand the movements and the environment structure. Keep track of the start and goal locations, and the current location in the environment as you watch the walkthrough. Then plan a shortcut route that takes you to the goal while avoiding unnecessary detours. Think step by step."
            ]
        else:
            raise ValueError(
                f"Unable to process walkthrough_key = {self.walkthrough_key}"
            )
        for frame in walkthrough_frames:
            frame_encoded = get_image_as_message(
                image=frame, model_name=self.model_name, image_detail=self.image_detail
            )
            output.append(frame_encoded)
        return output

    def convert_response_to_action(self, response_txt: str) -> str:
        try:
            re_out = re.search(r'{\s+"action":\s*"(.*)"\s+}', response_txt)
            pred_action = re_out.groups()[0]
        except Exception as e:
            print(f"Unable to parse predictions. Got exception {e}")
            pred_action = None
        return pred_action
