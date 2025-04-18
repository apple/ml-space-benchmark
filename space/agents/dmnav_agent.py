# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import re
from typing import Union

import numpy as np

from space.registry import register_agent
from space.utils.common import get_image_as_message
from space.agents.basenav_agent import Base_Navigation_Agent


IMAGE_TASK_PROMPT_ROUTE_FOLLOWING = """You are playing a game in a 2D world. You will be shown a video of the shortest route from an initial position to a goal. You must look at the video and understand the 2D world structure and the route taken. Then, you will be placed in the 2D world at the same initial position. You must navigate from the initial position to the goal using the same route shown in the video, as quickly as possible. Below, you will find sections highlighting more details about the 2D world and the task. You can refer to these for more information.

2D WORLD:
The world consists of the following.
* black cells: these are obstacles, i.e., you cannot move over them
* blue cells: these are navigable spaces, i.e., you can move over them

Some blue cells contain landmarks, which are red circles filled with a text character. These are important as they will allow you to better understand the world and locate yourself. Your position will be marked using a yellow square.

OBSERVATIONS:
The images are recorded from a birds-eye view of the 2D world. The images capture a local neighborhood surrounding your current position in the world, i.e., you will always remain at the center of the image while the world changes around you.

GOAL:
You will be asked to navigate to a goal landmark. You must find the goal in the 2D world by repeating the path shown in the video. Once you find it, move to the location of the goal till you are standing on the landmark and then execute a stop action.

ACTIONS:
You have four actions available.

up: move up by one unit cell
down: move down by one unit cell
left: move left by one unit cell
right: move right by one unit cell
stop: ends the current task. Issue this action only if you think you have reached the goal. If you haven't reached the goal, this action will result in a navigation failure that cannot be recovered from.

STOP CRITERIA:
Before executing stop, you must ensure that you've "reached" the goal correctly. To reach a goal, you have to move to the cell containing the goal landmark. Then execute the stop action.

RESPONSE FORMAT:
Respond in the following format:

Reasoning: <text explanation string in one or two short sentences --- provide all your explanations and inner thoughts here - avoid verbosity and be concise>
Intent: <state your intent in one short sentence, i.e., what you are trying to achieve>
Then provide the final action to take in a json formatted string.
```
{
    "action": <action name -- must be one of up, down, left, right>
}
```
"""

IMAGE_TASK_PROMPT_NOVEL_SHORTCUTS = """You are playing a game in a 2D world. You will be shown a video of some route from an initial position to a goal. You must look at the video and understand the 2D world structure and remember the locations of the start and the goal. The video may show a long-winded route from the start to the goal with unnecessary detours. Based on the world structure, you must identify a faster route to the goal. Then, you will be placed in the 2D world at the same initial position. You must navigate from the initial position to the goal using your identified shortest route as quickly as possible. Below, you will find sections highlighting more details about the 2D world and the task. You can refer to these for more information.

2D WORLD:
The world consists of the following.
* black cells: these are obstacles, i.e., you cannot move over them
* blue cells: these are navigable spaces, i.e., you can move over them

Some blue cells contain landmarks, which are red circles filled with a text character. These are important as they will allow you to better understand the world and locate yourself. Your position will be marked using a yellow square.

OBSERVATIONS:
The images are recorded from a birds-eye view of the 2D world. The images capture a local neighborhood surrounding your current position in the world, i.e., you will always remain at the center of the image while the world changes around you.

GOAL:
You will be asked to navigate to a goal landmark. You must find the goal in the 2D world by identifying the shortest path based your your experience from the video. Once you find it, move to the location of the goal till you are standing on the landmark and then execute a stop action.

ACTIONS:
You have four actions available.

up: move up by one unit cell
down: move down by one unit cell
left: move left by one unit cell
right: move right by one unit cell
stop: ends the current task. Issue this action only if you think you have reached the goal. If you haven't reached the goal, this action will result in a navigation failure that cannot be recovered from.

STOP CRITERIA:
Before executing stop, you must ensure that you've "reached" the goal correctly. To reach a goal, you have to move to the cell containing the goal landmark. Then execute the stop action.

RESPONSE FORMAT:
Respond in the following format:

Reasoning: <text explanation string in one or two short sentences --- provide all your explanations and inner thoughts here - avoid verbosity and be concise>
Intent: <state your intent in one short sentence, i.e., what you are trying to achieve>
Then provide the final action to take in a json formatted string.
```
{
    "action": <action name -- must be one of up, down, left, right>
}
```
"""

TEXT_TASK_PROMPT_ROUTE_FOLLOWING = """You are playing a game in a 2D text world. The console screen of the game is represented as a comma-separated text array. You will be shown a sequence of console screen recordings that demonstrate the shortest route from an initial position to a goal. You must look at the sequence and understand the 2D text world structure and the route taken. Then, you will be placed in the 2D text world at the same initial position. You must navigate from the initial position to the goal using the same route shown in the screen recording sequence, as quickly as possible. Below, you will find sections highlighting more details about the 2D text world and the task. You can refer to these for more information.

2D TEXT WORLD:
The console of the game is represented as a comma-separated text array. Obstacles are represented using 0, i.e., you cannot move over them. Navigable spaces that you can move over are represented using 1. Some navigable spaces have landmarks represented as an ascii character (A - Z). These are also navigable spaces and are just labeled with an ascii character. These landmarks are important to remember. You will always be located at the center of the array with your position highlighted using the "*" character.

OBSERVATIONS:
The images are recorded from a birds-eye view of the 2D world. The images capture a local neighborhood surrounding your current position in the world, i.e., you will always remain at the center of the image while the world changes around you.

GOAL:
You will be asked to navigate to a goal landmark. You must find the goal in the 2D text world by repeating the path shown in the console screen recording sequence. Once you find it, move to the location of the goal till you are standing on the landmark and then execute a stop action.

ACTIONS:
You have four actions available.

up: move up by one unit cell
down: move down by one unit cell
left: move left by one unit cell
right: move right by one unit cell
stop: ends the current task. Issue this action only if you think you have reached the goal. If you haven't reached the goal, this action will result in a navigation failure that cannot be recovered from.

STOP CRITERIA:
Before executing stop, you must ensure that you've "reached" the goal correctly. To reach a goal, you have to move to the cell containing the goal landmark. Then execute the stop action.

RESPONSE FORMAT:
Respond in the following format:

Reasoning: <text explanation string in one or two short sentences --- provide all your explanations and inner thoughts here - avoid verbosity and be concise>
Intent: <state your intent in one short sentence, i.e., what you are trying to achieve>
Then provide the final action to take in a json formatted string.
```
{
    "action": <action name -- must be one of up, down, left, right>
}
```
"""

TEXT_TASK_PROMPT_NOVEL_SHORTCUTS = """You are playing a game in a text 2D world. The console screen of the game is represented as a comma-separated text array. You will be shown a sequence of console screen recordings that demonstrates a route from an initial position to a goal. You must look at the sequence and understand the 2D text world structure and remember the locations of the start and the goal. The recordings may show a long-winded route from the start to the goal with unnecessary detours. Based on the world structure, you must identify a faster route to the goal. Then, you will be placed in the 2D text world at the same initial position. You must navigate from the initial position to the goal using your identified shortest route as quickly as possible. Below, you will find sections highlighting more details about the 2D text world and the task. You can refer to these for more information.

2D TEXT WORLD:
The console of the game is represented as a comma-separated text array. Obstacles are represented using 0, i.e., you cannot move over them. Navigable spaces that you can move over are represented using 1. Some navigable spaces have landmarks represented as an ascii character (A - Z). These are also navigable spaces and are just labeled with an ascii character. These landmarks are important to remember. You will always be located at the center of the array with your position highlighted using the "*" character.

OBSERVATIONS:
The images are recorded from a birds-eye view of the 2D world. The images capture a local neighborhood surrounding your current position in the world, i.e., you will always remain at the center of the image while the world changes around you.

GOAL:
You will be asked to navigate to a goal landmark. You must find the goal in the 2D text world by identifying the shortest path based your your experience from the screen recording sequence. Once you find it, move to the location of the goal till you are standing on the landmark and then execute a stop action.

ACTIONS:
You have four actions available.

up: move up by one unit cell
down: move down by one unit cell
left: move left by one unit cell
right: move right by one unit cell
stop: ends the current task. Issue this action only if you think you have reached the goal. If you haven't reached the goal, this action will result in a navigation failure that cannot be recovered from.

STOP CRITERIA:
Before executing stop, you must ensure that you've "reached" the goal correctly. To reach a goal, you have to move to the cell containing the goal landmark. Then execute the stop action.

RESPONSE FORMAT:
Respond in the following format:

Reasoning: <text explanation string in one or two short sentences --- provide all your explanations and inner thoughts here - avoid verbosity and be concise>
Intent: <state your intent in one short sentence, i.e., what you are trying to achieve>
Then provide the final action to take in a json formatted string.
```
{
    "action": <action name -- must be one of up, down, left, right>
}
```
"""


class Base_DiscreteMap_Nav(Base_Navigation_Agent):
    def get_action(self, obs: Union[np.ndarray, str]):
        new_message = self.get_observation_prompt(obs)
        delta_txt = "What action do you select next? The available actions are up, down, left, right and stop."
        if isinstance(new_message, str):
            new_message += "\n" + delta_txt
        else:
            new_message.append(delta_txt)

        response_txt = self.handle_model_response(new_message)
        ############################################################################################
        # Clean-up context history
        # Remove explanatory text content of last user message "What actions do you select next? ...."
        assert self.dialog.history[-2]["role"] == "user"
        self.dialog.history[-2] = {
            "role": "user",
            "content": self.get_observation_prompt(obs),
        }
        ############################################################################################
        action = self.convert_response_to_action(response_txt)
        if action is None:
            # Turn left if no action was parseable from the model
            action = "up"

        return action

    def convert_response_to_action(self, response_txt: str) -> str:
        try:
            re_out = re.search(r'{\s+"action":\s*"(.*)"\s+}', response_txt)
            pred_action = re_out.groups()[0]
            if pred_action not in ["up", "down", "left", "right", "stop"]:
                print(
                    f"Obtained invalid action: {pred_action} from model. Replacing it with `up`."
                )
                pred_action = "up"
        except Exception as e:
            print(f"Unable to parse predictions. Got exception {e}")
            pred_action = None
        return pred_action

    def get_observation_prompt(self, obs: Union[np.ndarray, str]):
        raise NotImplementedError


@register_agent
class DiscreteMapImage_Nav_Agent(Base_DiscreteMap_Nav):
    def get_observation_prompt(self, obs: np.ndarray):
        return [
            "Here is the local view of your surroundings in the 2D world. You are at the center of this view.",
            get_image_as_message(
                image=obs, model_name=self.model_name, image_detail=self.image_detail
            ),
        ]

    def get_walkthrough_prompt(self, walkthrough_obs: list[np.ndarray]):
        if self.walkthrough_key == "shortestpath":
            output = [
                "Here is sequence of video frames recorded in the 2D world. This demonstrates the route you need to repeat. Analyze the video to understand the movements and the world structure. Take a note of all the details needed to help you repeat this route when navigating next. Think step by step."
            ]
        elif self.walkthrough_key == "walkthrough":
            output = [
                "Here is the sequence of video frames recorded in the 2D world. This demonstrates a suboptimal route from the start to some goal location. Analyze the video to understand the movements and the world structure. Keep track of the start and goal locations, and the current location in the world as you watch the video. Then plan a shortcut route that takes you to the goal while avoiding any unnecessary detours. Think step by step."
            ]
        else:
            raise ValueError(
                f"Unable to process walkthrough_key = {self.walkthrough_key}"
            )
        for frame in walkthrough_obs:
            frame_encoded = get_image_as_message(
                image=frame, model_name=self.model_name, image_detail=self.image_detail
            )
            output.append(frame_encoded)
        return output

    def get_goal_prompt(self, goal_desc: str, *args, **kwargs):
        prompt = f"Now, you must navigate to the goal based on your knowledge of the 2D world you obtained from the video. Here is the goal description: {goal_desc}"

        return prompt

    def set_task_prompt(self, walkthrough_key: str):
        assert walkthrough_key in ["shortestpath", "walkthrough"]
        if walkthrough_key == "shortestpath":
            # Route following experiment
            self.task_prompt = IMAGE_TASK_PROMPT_ROUTE_FOLLOWING
        else:
            # Novel shortcuts experiment
            self.task_prompt = IMAGE_TASK_PROMPT_NOVEL_SHORTCUTS


@register_agent
class DiscreteMapText_Nav_Agent(Base_DiscreteMap_Nav):
    def get_observation_prompt(self, obs: str):
        return obs

    def get_walkthrough_prompt(self, walkthrough_obs: list[str]):
        if self.walkthrough_key == "shortestpath":
            output = [
                "Here is the sequence of console screen recordings taken in the 2D text world. This demonstrates the route you need to repeat. Analyze the sequence to understand the movements and the world structure. Take a note of all the details needed to help you repeat this route when navigating next. Think step by step."
            ]
        elif self.walkthrough_key == "walkthrough":
            output = [
                "Here is the sequence of console screen recordings taken in the 2D text world. This demonstrates a suboptimal route from the start to some goal location. Analyze the sequence to understand the movements and the world structure. Keep track of the start and goal locations, and the current location in the world as you study the sequence. Then plan a shortcut route that takes you to the goal while avoiding any unnecessary detours. Think step by step."
            ]
        else:
            raise ValueError(
                f"Unable to process walkthrough_key = {self.walkthrough_key}"
            )
        for i, text in enumerate(walkthrough_obs):
            output.append(
                f"### Console screen recorded at time = {i}\n\n```\n{text}\n```"
            )
        output = "\n\n".join(output)
        return output

    def get_goal_prompt(self, goal_desc: str, *args, **kwargs):
        prompt = f"Now, you must navigate to the goal based on your knowledge of the 2D text world you obtained from the sequence of console screen recordings. Here is the goal description: {goal_desc}"

        return prompt

    def set_task_prompt(self, walkthrough_key: str):
        assert walkthrough_key in ["shortestpath", "walkthrough"]
        if walkthrough_key == "shortestpath":
            # Route following experiment
            self.task_prompt = TEXT_TASK_PROMPT_ROUTE_FOLLOWING
        else:
            # Novel shortcuts experiment
            self.task_prompt = TEXT_TASK_PROMPT_NOVEL_SHORTCUTS
