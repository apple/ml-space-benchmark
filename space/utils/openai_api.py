# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import copy
import json
import os
import time
from typing import Any

from openai import OpenAI
from space.utils.common import convert_content_to_str


def setup_llm_client(model_name, host_port: str = "8000"):
    if model_name.startswith("gpt-"):
        assert "OPENAI_API_KEY" in os.environ
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = None
    else:
        api_key = "EMPTY"
        base_url = f"http://localhost:{host_port}/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


class Dialog:
    def __init__(self, log_writer=None):
        self.history = []
        self.log_writer = log_writer
        self.images_save_dir = None
        self.log_dir = None
        if log_writer is not None:
            self.log_dir = os.path.dirname(self.log_writer.file_name)
            self.log_count = 0
            self.images_save_dir = os.path.dirname(self.log_writer.file_name)
            os.makedirs(os.path.join(self.images_save_dir, "images"), exist_ok=True)

    def add_system_message(self, **kwargs):
        self.history.append({"role": "system", **kwargs})
        self.log_to_file(
            "system", convert_content_to_str(kwargs["content"], self.images_save_dir)
        )

    def add_user_message(self, **kwargs):
        self.history.append({"role": "user", **kwargs})
        self.log_to_file(
            "user", convert_content_to_str(kwargs["content"], self.images_save_dir)
        )

    def add_assistant_message(self, **kwargs):
        self.history.append({"role": "assistant", **kwargs})
        self.log_to_file(
            "assistant",
            convert_content_to_str(kwargs["content"], self.images_save_dir),
        )

    def add_inner_thoughts(self, **kwargs):
        if kwargs["content"] is not None:
            self.log_to_file(
                "assistant (inner thoughts)",
                convert_content_to_str(kwargs["content"], self.images_save_dir),
                bold_italics_code="bi",
            )

    def log_to_file(self, role, content_str, bold_italics_code="b"):
        if self.log_writer is not None:
            self.log_writer.new_paragraph(
                f"{role.upper()}", bold_italics_code=bold_italics_code
            )
            self.log_writer.new_paragraph("\n" + content_str + "\n")
            self.log_writer.create_md_file()

    def log_token_usage(
        self, prompt_tokens: int, completion_tokens: int, total_cost: int = None
    ):
        if self.log_writer is not None:
            self.log_writer.write("\n")
            self.log_writer.write(
                "====> Token usage: prompt tokens: {}, completion_tokens: {}".format(
                    prompt_tokens, completion_tokens
                )
            )
            if total_cost is not None:
                self.log_writer.write("\n")
                self.log_writer.write(f"====> Total cost: ${total_cost:.3f}")
            self.log_writer.write("\n\n")
            self.log_writer.create_md_file()

    def log_response_time(self, time_taken: float):
        if self.log_writer is not None:
            self.log_writer.write("\n")
            self.log_writer.write(f"====> Response time (sec): {time_taken:.3f}")
            self.log_writer.write("\n\n")
            self.log_writer.create_md_file()

    @property
    def dialog(self):
        return copy.deepcopy(self.history)

    def delete_last_message(self):
        del self.history[-1]

    def clear_history(self):
        self.history = []

    def clone(self):
        dialog_clone = Dialog(self.log_writer)
        dialog_clone.history = copy.deepcopy(self.history)
        return dialog_clone

    def write_dialog(self):
        if self.log_writer is not None:
            save_path = os.path.join(self.log_dir, f"dialog_{self.log_count:05d}.json")
            with open(save_path, "w") as fp:
                json.dump(self.history, fp)
            save_path = os.path.join(self.log_dir, f"dialog_{self.log_count:05d}.txt")
            with open(save_path, "w") as fp:
                for h in self.history:
                    content_str = convert_content_to_str(
                        h["content"], save_dir=None, ignore_images=True
                    )
                    role_str = h["role"].upper()
                    fp.write(f"{role_str}: {content_str}\n\n")
            self.log_count += 1
        else:
            print(
                "WARNING: Dialog object does not have a log_writer, so write_dialog() failed..."
            )


# Function to get response from model
def get_model_response(
    client: Any,
    dialog: list[Any],
    model_name: str = "gpt-4-vision-preview",
    temperature: float = 0.5,
    max_tokens: int = 1000,
    frequency_penalty: float = 0.0,
    max_retries: int = 10,
    sleep_secs: int = 60,
    verbose: bool = True,
    **kwargs,
):
    num_retries = 0
    start_time = time.time()
    excptn_info = {}
    while num_retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=dialog,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                **kwargs,
            )
            response_txt = response.choices[0].message.content
            token_counts = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
            }
            break
        except Exception as excptn:
            num_retries += 1
            if num_retries >= max_retries:
                if verbose:
                    print(excptn)
                excptn_info = {
                    "exception_message": excptn.message,
                    "exception_type": excptn.type,
                    "exception_code": excptn.code,
                }
            time.sleep(sleep_secs)

    if num_retries >= max_retries:
        if verbose:
            print(f"===> Failed after {max_retries} retries")
        response_txt = None
        token_counts = {}

    time_taken = time.time() - start_time
    output = {
        "text": response_txt,
        **token_counts,
        "response_time": time_taken,
        **excptn_info,
    }
    return output
