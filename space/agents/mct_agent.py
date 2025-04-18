# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
import re
import time
from abc import ABC
from typing import Any, Union

import numpy as np
from mdutils.mdutils import MdUtils

from space.registry import register_agent
from space.utils.common import get_image_as_message
from space.utils.openai_api import (
    Dialog,
    setup_llm_client,
)

from space.utils.claude_api import (
    Dialog as DialogClaude,
    setup_llm_client as setup_llm_client_claude,
)
from space.utils.model import get_model_response
from space.utils.common import count_images_in_query


VISION_TASK_PROMPT = """You are a sentient living creature capable navigating in mazes, planning, and spatial reasoning. You are playing a Pacman-style maze game. You start at some random position in the maze. You must escape the maze as quickly as possible to reach the goal. You are given the game screen that shows the following:
* maze structure - blue is obstacle space, black is navigable space. You can only move on black spaces. You cannot move through blue spaces.
* your current position - yellow square
* goal position - red circle

Below the screen, a status message might appear indicating that you collided into a wall after your previous action.

Actions available: You can take five possible actions.
* left - move left from your current position by one step
* right - move right from your current position by one step
* up - move up from your current position by one step
* down - move down from your current position by one step
* stop - issue this action only after you have reached the goal position. If you execute it prematurely, you will fail. If you do not execute it after reaching the goal, you will again fail.

Response format: Respond in the following format.

<text explanation string - explain your reasoning concisely>
<next, provide a json formatted output with the next action>
```
{
    "action": "<action>"
}
```
"""

TEXT_TASK_PROMPT = """You are a sentient living creature capable navigating in mazes, planning, and spatial reasoning. You are playing a text-based maze game. You start at some random position in the maze. You must escape the maze as quickly as possible to reach the goal. You are given a 2D array representing the maze, which contains the following:
* maze structure - 0 is obstacle space, 1 is navigable space. You can only move on 1s (i.e., navigable spaces). You cannot move through 0s (i.e., obstacles).
* your current position - marked as A
* goal position - marked as G

Goal and current positions are always navigable spaces.

Actions available: You can take five possible actions.
* left - move left from your current position by one step
* right - move right from your current position by one step
* up - move up from your current position by one step
* down - move down from your current position by one step
* stop - issue this action only after you have reached the goal position. If you execute it prematurely, you will fail. If you do not execute it after reaching the goal, you will again fail.

Response format: Respond in the following format.

<Think step-by-step about what action to take next. Be concise.>
<next, provide a json formatted output with the next action>
```
{
    "action": "<action>"
}
```
"""


@register_agent
class MCT_Agent(ABC):
    def __init__(
        self,
        model_name: str,
        host_port: str,
        save_dir: str,
        description_type: str,
        image_detail: str = "low",
        max_new_tokens: int = 2048,
        max_history_length: int = 20,
        completion_cost_per_mil: float = 0.0,
        prompt_cost_per_mil: float = 0.0,
        supports_system_prompt: bool = True,
        max_context_tokens: int = 8192,
        context_truncation_factor: float = 0.9,
        max_images_per_query: int = -1,
        **kwargs,
    ):
        assert description_type in ["image", "text"]
        self.model_name = model_name
        self.host_port = host_port
        self.save_dir = save_dir
        self.description_type = description_type
        self.image_detail = image_detail
        self.max_new_tokens = max_new_tokens
        self.max_history_length = max_history_length
        self.completion_cost_per_mil = completion_cost_per_mil
        self.prompt_cost_per_mil = prompt_cost_per_mil
        self.supports_system_prompt = supports_system_prompt
        self.max_context_tokens = max_context_tokens
        self.context_truncation_factor = context_truncation_factor
        self.max_images_per_query = max_images_per_query
        self.writer = None
        self.dialog = None
        self.completion_tokens = None
        self.prompt_tokens = None

        if self.model_name.startswith("claude"):
            self.client = setup_llm_client_claude(self.model_name)
        else:
            self.client = setup_llm_client(self.model_name, self.host_port)

        if self.description_type == "image":
            self.task_prompt = VISION_TASK_PROMPT
        else:
            self.task_prompt = TEXT_TASK_PROMPT

    def reset(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = MdUtils(
            file_name=os.path.join(self.save_dir, "transcript"),
            title="SPACE MCT",
        )
        if self.model_name.startswith("claude"):
            self.dialog = DialogClaude(self.writer)
        else:
            self.dialog = Dialog(self.writer)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        # Setup system prompt
        if self.supports_system_prompt:
            self.dialog.add_system_message(content=self.task_prompt)
        else:
            self.dialog.add_user_message(content=self.task_prompt)
            self.dialog.add_assistant_message(
                content="Okay, I understand. Let's begin!"
            )

    def get_action(self, obs: Union[str, np.ndarray]):
        content = self.get_user_message(obs)
        response_txt = self.handle_model_response(content)
        pred_action = self.parse_answer_from_response(response_txt)
        return pred_action

    def get_user_message(self, obs: Union[np.ndarray, str]):
        if self.description_type == "image":
            assert isinstance(obs, np.ndarray)
            content = [
                "Here is the current state of the maze.",
                get_image_as_message(
                    image=obs,
                    model_name=self.model_name,
                    image_detail=self.image_detail,
                ),
            ]
            content.append(
                "Think step-by-step about how to reach the goal. What action do you take next?"
            )
        else:
            assert isinstance(obs, str)
            content = obs
            content += "Think step-by-step about how to reach the goal. What action do you take next?"
        return content

    def postprocess_response(self, text: str):
        if self.model_name.startswith("mistralai"):
            text = text.split("[/INST]")[-1].strip()
        return text

    def perform_model_response_loop(self, max_retries: int = 10, sleep_sec: int = 60):
        n_retries = 0
        response = None
        while n_retries <= max_retries:
            response = get_model_response(
                self.client,
                self.dialog.dialog,
                self.model_name,
                max_tokens=self.max_new_tokens,
                max_retries=1,
            )
            excptn_type = response.get("exception_type", None)
            excptn_code = response.get("exception_code", None)
            excptn_message = response.get("exception_message", None)
            # Check for out-of-context error
            ooc_error = (
                excptn_type == "BadRequestError"
                and excptn_code == 400
                and "maximum context length" in excptn_message
            )
            if ooc_error:
                # Out-of-context error => restrict dialog and retry
                self.restrict_dialog_history()
            elif excptn_message is None and response["text"] is not None:
                # No errors => finish
                break
            else:
                # Some other error (e.g., rate limits) => retry after sleep_sec
                time.sleep(sleep_sec)
            n_retries += 1
        if n_retries >= max_retries and response is None:
            print(
                f"Failed after {n_retries} retries due to following error: {response['excptn_message']}"
            )

        return response

    def handle_model_response(self, content: Union[list[Any], str]):
        n_images = count_images_in_query(content)
        if self.max_images_per_query >= 1 and n_images > self.max_images_per_query:
            c_subset = []
            n_images_subset = 0
            for c in content:
                if isinstance(c, dict) and c.get("type", None) == "image":
                    n_images_subset += 1
                c_subset.append(c)
                if n_images_subset == self.max_images_per_query:
                    c_subset.append(
                        "More information will be provided in the next image. Do not say anything. Please wait before responding."
                    )
                    self.dialog.add_user_message(content=c_subset)
                    self.dialog.add_assistant_message(
                        content="I understand. I'll continue to wait for your next message before providing any analysis or response."
                    )
                    # Reset
                    n_images_subset = 0
                    c_subset = []
            c_subset.append(
                "This marks the end of my message to you. Please respond now."
            )
            self.dialog.add_user_message(content=c_subset)
        else:
            self.dialog.add_user_message(content=content)
        response = self.perform_model_response_loop()
        ##########################################################
        prompt_tokens = response["prompt_tokens"]
        completion_tokens = response["completion_tokens"]
        ##########################################################
        response_txt = self.postprocess_response(response["text"])
        self.dialog.add_assistant_message(content=response_txt)
        self.dialog.write_dialog()
        ############################################################################################
        # Smart context handling
        if response is not None:
            request_tokens = prompt_tokens + completion_tokens
            if (
                request_tokens + self.max_new_tokens
                >= self.context_truncation_factor * self.max_context_tokens
            ):
                # Restrict dialog history
                self.restrict_dialog_history()
        ############################################################################################
        # Truncate history if needed
        if self.max_history_length > 0:
            history_len = len(self.dialog.history)
            if self.supports_system_prompt:
                history_len -= 1
            else:
                history_len -= 2
            if history_len > self.max_history_length:
                n_steps = (history_len - self.max_history_length) // 2
                for _ in range(n_steps):
                    self.restrict_dialog_history()
        ############################################################################################
        # Track token usage
        self.completion_tokens += completion_tokens
        self.prompt_tokens += prompt_tokens
        total_cost = (
            self.completion_tokens * self.completion_cost_per_mil / 1.0e6
            + self.prompt_tokens * self.prompt_cost_per_mil / 1.0e6
        )
        self.dialog.log_token_usage(
            self.prompt_tokens, self.completion_tokens, total_cost
        )
        # Log time taken
        self.dialog.log_response_time(response["response_time"])
        return response_txt

    def parse_answer_from_response(self, response_txt: str):
        try:
            re_out = re.search(r'{\s*"action":\s*"(.*)"\s*}', response_txt)
            pred_action = re_out.groups()[0]
            if pred_action not in ["up", "left", "right", "down", "stop"]:
                print(
                    f".... Invalid action predicted: `{pred_action}`. Replacing it with action `up`."
                )
                pred_action = "up"
        except Exception as e:
            print(
                f"Unable to parse predictions from '{response_txt}'. Got exception {e}"
            )
            pred_action = "up"
        return pred_action

    def restrict_dialog_history(self):
        if len(self.dialog.history) <= 1:
            return
        if self.supports_system_prompt:
            task_context = self.dialog.history[:1]
            dialog_history = self.dialog.history[1:]
        else:
            task_context = self.dialog.history[:2]
            dialog_history = self.dialog.history[2:]
        self.dialog.history = task_context + dialog_history[2:]

    def get_eval_cost(self):
        total_cost = (
            self.completion_tokens * self.completion_cost_per_mil / 1.0e6
            + self.prompt_tokens * self.prompt_cost_per_mil / 1.0e6
        )
        return {
            "total_cost": total_cost,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
        }
