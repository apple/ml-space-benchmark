# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
import re
import time
import numpy as np

from typing import Any, Union
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


VISION_TASK_PROMPT = """You are playing the Cambridge Spatial Working Memory game. You will be shown a screen with blue boxes. A treasure is hidden in one of the blue boxes. You must identify the box containing the treasure, which is shown as an yellow square. Once you find a treasure, it will be collected and placed in the "Treasures collected" section shown below the image. A new treasure will be hidden in one of the other boxes where the treasure did not appear before. You must again find the new treasure. This process is repeated till you find all treasures placed in each of the blue boxes once. Note: The treasure will never appear in a box where it had already been placed.

Each turn, there are randomly selected numbers associated with each box. These numbers are meant to aid you with communication, i.e., specify what box you want to open in that turn. However, these numbers will change after every turn. So do NOT associate boxes with numbers over the long term. The number identity of a box can change any time. Therefore, you must remember the boxes based on their spatial positions and not the numbers.

RESPONSE FORMAT:
Think step-by-step about where the treasure might be based on your past actions. After that, indicate the box you want to open in the following json format:
```
{
    "action": <box integer index>
}
```"""

TEXT_TASK_PROMPT = """You are playing the Cambridge Spatial Working Memory game. You will be shown an array with integers. 0 represents empty locations. Locations numbered 1 - 9 represent boxes. A treasure is hidden in one of the boxes. You must identify the box containing the treasure. Once you find a treasure, the location will be momentarily shown as a "T" indicating that the treasure was found. The treasure is then collected and a new treasure will be hidden in one of the other boxes where the treasure did not appear before. You must then find the new treasure. This process is repeated till you find all treasures placed in each of the boxes once. Note: The treasure will never appear in a box where it had already been placed.

While the boxes are represented using integers from 1 - 9, the true identity of the box is its location (row, column) in the array. The box location is always fixed (i.e., the boxes will not move and the number of boxes will not change). However, each turn, the integer id associated with the box will change randomly. These integer ids are meant to aid you with communication, i.e., specify what box you want to open in that turn. However, these numbers will change after every turn. So do NOT associate boxes with numbers over the long term. The number identity of a box can change any time. Therefore, you must remember the boxes based on their spatial positions and not the numbers.

RESPONSE FORMAT:
Think step-by-step about where the treasure might be based on your past actions. After that, indicate the box you want to open in the following json format:
```
{
    "action": <box integer index>
}
```"""


@register_agent
class CSWM_Agent(object):
    def __init__(
        self,
        model_name: str,
        host_port: str,
        save_dir: str,
        task_mode: str,
        image_detail: str = "low",
        max_new_tokens: int = 2048,
        max_history_length: int = -1,
        completion_cost_per_mil: float = 0.0,
        prompt_cost_per_mil: float = 0.0,
        supports_system_prompt: bool = True,
        max_context_tokens: int = 8192,
        context_truncation_factor: float = 0.9,
        max_images_per_query: int = -1,
        **kwargs,
    ):
        assert task_mode in ["vision", "text"]
        self.model_name = model_name
        self.host_port = host_port
        self.save_dir = save_dir
        self.task_mode = task_mode
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

        if self.task_mode == "vision":
            self.task_prompt = VISION_TASK_PROMPT
        else:
            self.task_prompt = TEXT_TASK_PROMPT

    def reset(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = MdUtils(
            file_name=os.path.join(self.save_dir, "transcript"),
            title="SPACE CSWM",
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

    def get_action(self, obs: Union[np.ndarray, str]):
        content = self.get_user_message(obs)
        response_txt = self.handle_model_response(content)
        pred_action = self.parse_answer_from_response(response_txt)
        return pred_action

    def get_user_message(self, obs: Union[np.ndarray, str]):
        if self.task_mode == "vision":
            assert isinstance(obs, np.ndarray)
            content = [
                "Here is the current state of the game. You must find the next treasure. Note that the numbers of the boxes have changed, but the box locations are fixed. Decide which box you want to open next, and then use the number associated with the box as the action.",
                get_image_as_message(
                    image=obs,
                    model_name=self.model_name,
                    image_detail=self.image_detail,
                ),
            ]
        else:
            assert isinstance(obs, str)
            content = obs
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
            re_out = re.search(r'{\s*"action":\s*(.*)\s*}', response_txt)
            pred_action = int(re_out.groups()[-1])
        except Exception as e:
            print(
                f"Unable to parse predictions from '{response_txt}'. Got exception {e}"
            )
            pred_action = 0 if self.task_mode == "vision" else 1
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
