from typing import Any
from func_timeout import FunctionTimedOut, func_timeout
from space.utils.claude_api import get_model_response as get_model_response_claude
from space.utils.openai_api import get_model_response as get_model_response_openai


def get_model_response(
    client: Any,
    dialog: list[Any],
    model_name: str,
    *args,
    max_retries: int = 10,
    max_response_wait_secs_per_retry: float = 300.0,
    **kwargs,
):
    max_response_wait_secs = max_retries * max_response_wait_secs_per_retry
    try:
        if model_name.startswith("claude"):
            response = func_timeout(
                max_response_wait_secs,
                get_model_response_claude,
                args=(client, dialog, model_name, *args),
                kwargs=kwargs,
            )
        else:
            response = func_timeout(
                max_response_wait_secs,
                get_model_response_openai,
                args=(client, dialog, model_name, *args),
                kwargs=kwargs,
            )
    except FunctionTimedOut:
        print(f"get_model_response() timed out after {max_response_wait_secs} secs!")
        response = {
            "text": "timed out",
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "response_time": max_response_wait_secs,
        }
    return response
