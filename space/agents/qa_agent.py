# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import json
import os
import re
from typing import Any, Union

from mdutils.mdutils import MdUtils
from space.registry import register_agent

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


def is_int(text: str):
    try:
        int(text)
    except Exception:
        return False
    else:
        return True


@register_agent
class QA_Agent(object):
    def __init__(
        self,
        model_name: str,
        host_port: str,
        save_dir: str,
        image_detail: str = "low",
        max_new_tokens: int = 2048,
        completion_cost_per_mil: float = 0.0,
        prompt_cost_per_mil: float = 0.0,
        subsampling_factor: int = 1,
        max_images_per_query: int = -1,
        **kwargs,
    ):
        self.model_name = model_name
        self.host_port = host_port
        self.save_dir = save_dir
        self.image_detail = image_detail
        self.max_new_tokens = max_new_tokens
        self.completion_cost_per_mil = completion_cost_per_mil
        self.prompt_cost_per_mil = prompt_cost_per_mil
        self.subsampling_factor = subsampling_factor
        self.max_images_per_query = max_images_per_query
        self.writer = None
        self.dialog = None
        self.completion_tokens = None
        self.prompt_tokens = None
        if self.model_name.startswith("claude"):
            self.client = setup_llm_client_claude(self.model_name)
        else:
            self.client = setup_llm_client(self.model_name, self.host_port)

    def reset(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = MdUtils(
            file_name=os.path.join(self.save_dir, "transcript"),
            title="Question-answering task",
        )
        if self.model_name.startswith("claude"):
            self.dialog = DialogClaude(self.writer)
        else:
            self.dialog = Dialog(self.writer)
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def update_save_dir(self, save_dir: str):
        self.save_dir = save_dir

    def get_prediction(self, question_content: Union[list[Any], str], answer: Any):
        question_content = self.preprocess_question(question_content)
        n_images = count_images_in_query(question_content)
        if self.max_images_per_query >= 1 and n_images > self.max_images_per_query:
            qc_subset = []
            n_images_subset = 0
            for q in question_content:
                if isinstance(q, dict) and q.get("type", None) == "image":
                    n_images_subset += 1
                qc_subset.append(q)
                if n_images_subset == self.max_images_per_query:
                    qc_subset.append(
                        "More information will be provided in the next image. Do not say anything. Please wait before responding."
                    )
                    self.dialog.add_user_message(content=qc_subset)
                    self.dialog.add_assistant_message(
                        content="I understand. I'll continue to wait for your next message before providing any analysis or response."
                    )
                    # Reset
                    n_images_subset = 0
                    qc_subset = []
            qc_subset.append(
                "This marks the end of my message to you. Please respond now."
            )
            self.dialog.add_user_message(content=qc_subset)
        else:
            self.dialog.add_user_message(content=question_content)

        response = get_model_response(
            self.client,
            self.dialog.dialog,
            self.model_name,
            max_tokens=self.max_new_tokens,
        )
        response_txt = self.postprocess_response(response["text"])
        self.dialog.add_assistant_message(content=response_txt)
        pred = self.parse_answer_from_response(response_txt)
        # Add GT answer and prediction for reference
        self.dialog.log_writer.write(
            f"\n\nGround-truth answer: {answer}, prediction: {pred}"
        )

        ############################################################################################
        # Log token usage
        self.completion_tokens += response["completion_tokens"]
        self.prompt_tokens += response["prompt_tokens"]
        total_cost = (
            self.completion_tokens * self.completion_cost_per_mil / 1.0e6
            + self.prompt_tokens * self.prompt_cost_per_mil / 1.0e6
        )
        self.dialog.log_token_usage(
            self.prompt_tokens, self.completion_tokens, total_cost
        )
        # Log time taken
        self.dialog.log_response_time(response["response_time"])
        ############################################################################################

        ############################################################################################
        # Remove prior messages from history
        self.dialog.delete_last_message()
        self.dialog.delete_last_message()
        ############################################################################################
        return pred

    def preprocess_question(self, question_content: Union[list[Any], str]):
        assert isinstance(question_content, str) or isinstance(question_content, list)
        if self.model_name.startswith("mistralai/Pixtral"):
            if isinstance(question_content, str):
                question_content = [{"type": "text", "text": question_content}]
            else:
                question_content_p = []
                for q in question_content:
                    if isinstance(q, str):
                        question_content_p.append({"type": "text", "text": q})
                    elif isinstance(q, dict):
                        question_content_p.append(q)
                    else:
                        raise ValueError(f"Unable to preprocess question content: {q}")
                question_content = question_content_p
        return question_content

    def postprocess_response(self, response_txt: str):
        if self.model_name.startswith("mistralai"):
            response_txt = response_txt.split("[/INST]")[-1].strip()
        return response_txt

    def parse_answer_from_response(self, text: str):
        outputs = re.findall(r"({.*?})", text, re.DOTALL)
        if len(outputs) == 0:
            print(f"Unable to parse answer from text:\n{text}")
            return None
        output = outputs[-1].strip()
        try:
            answer_dict = json.loads(output)
        except ValueError:
            print(f"Unable to decode json from text:\n{text}")
            answer = None
        else:
            if "answer" in answer_dict:
                answer = {"answer": answer_dict["answer"]}
            elif len(answer_dict) == 1:
                answer = {"answer": list(answer_dict.values())[0]}
            else:
                print(
                    f"Ambiguous response in text (dict without key `answer`):\n{text}"
                )
                answer = None
        if answer is not None and answer["answer"] is None:
            answer = None
        if answer is not None and is_int(answer["answer"]):
            answer["answer"] = int(answer["answer"])

        if answer is not None:
            answer = answer["answer"]
        return answer

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
