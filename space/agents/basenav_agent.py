# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
import numpy as np
import time

from abc import ABC
from typing import Any, Optional, Union
from mdutils.mdutils import MdUtils

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


class Base_Navigation_Agent(ABC):
    def __init__(
        self,
        model_name: str,
        host_port: str,
        save_dir: str,
        image_detail: str = "low",
        max_new_tokens: int = 2048,
        max_history_length: int = -1,
        completion_cost_per_mil: float = 0.0,
        prompt_cost_per_mil: float = 0.0,
        supports_system_prompt: bool = True,
        max_context_tokens: int = 8192,
        context_truncation_factor: float = 0.9,
        subsampling_factor: int = 1,
        max_images_per_query: int = -1,
        **kwargs,
    ):
        self.model_name = model_name
        self.host_port = host_port
        self.save_dir = save_dir
        self.image_detail = image_detail
        self.max_new_tokens = max_new_tokens
        self.max_history_length = max_history_length
        self.completion_cost_per_mil = completion_cost_per_mil
        self.prompt_cost_per_mil = prompt_cost_per_mil
        self.supports_system_prompt = supports_system_prompt
        self.max_context_tokens = max_context_tokens
        self.context_truncation_factor = context_truncation_factor
        self.subsampling_factor = subsampling_factor
        self.max_images_per_query = max_images_per_query
        self.writer = None
        self.walkthrough_key = None
        self.dialog = None
        self.completion_tokens = None
        self.prompt_tokens = None

        if self.model_name.startswith("claude"):
            self.client = setup_llm_client_claude(self.model_name)
        else:
            self.client = setup_llm_client(self.model_name, self.host_port)
        self.task_prompt = None

    def reset(self, walkthrough_key: str):
        assert walkthrough_key in ["shortestpath", "walkthrough"]
        self.walkthrough_key = walkthrough_key
        self.set_task_prompt(walkthrough_key)
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = MdUtils(
            file_name=os.path.join(self.save_dir, "transcript"),
            title="SPACE navigation",
        )
        if self.model_name.startswith("claude"):
            self.dialog = DialogClaude(self.writer)
        else:
            self.dialog = Dialog(self.writer)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.necessary_context_len = 0
        # Setup system prompt
        if self.supports_system_prompt:
            self.dialog.add_system_message(content=self.task_prompt)
            self.necessary_context_len += 1
        else:
            self.dialog.add_user_message(content=self.task_prompt)
            self.dialog.add_assistant_message(
                content="Okay, I understand. Let's begin!"
            )
            self.necessary_context_len += 2

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
            history_len = len(self.dialog.history) - self.necessary_context_len
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
        self.writer.create_md_file()
        return response_txt

    def postprocess_response(self, text: str):
        if self.model_name.startswith("mistralai"):
            text = text.split("[/INST]")[-1].strip()
        return text

    def initialize_with_walkthrough(
        self, walkthrough_obs: Union[list[np.ndarray], list[str]]
    ):
        if self.subsampling_factor > 1:
            walkthrough_obs = (
                [walkthrough_obs[0]]
                + walkthrough_obs[1 : -1 : self.subsampling_factor]
                + [walkthrough_obs[-1]]
            )
        prompt = self.get_walkthrough_prompt(walkthrough_obs)
        _ = self.handle_model_response(prompt)
        self.necessary_context_len += 2

    def initialize_with_goal(
        self, goal_desc: str, goal_img: Optional[np.ndarray] = None
    ):
        message = self.get_goal_prompt(goal_desc, goal_img)
        self.dialog.add_user_message(content=message)
        self.necessary_context_len += 1

    def set_task_prompt(self, walkthrough_key: str):
        raise NotImplementedError

    def get_goal_prompt(self, goal_desc: str, goal_img: Optional[np.ndarray] = None):
        raise NotImplementedError

    def get_action(self, obs: Union[np.ndarray, str]):
        raise NotImplementedError

    def get_walkthrough_prompt(
        self, walkthrough_obs: Union[list[np.ndarray], list[str]]
    ):
        raise NotImplementedError

    def restrict_dialog_history(self):
        if len(self.dialog.history) <= 1:
            return

        task_context = self.dialog.history[: self.necessary_context_len]
        dialog_history = self.dialog.history[self.necessary_context_len :]
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
