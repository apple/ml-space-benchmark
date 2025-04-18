# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Any, Optional, Union

import os
import cv2
import base64
import time
import random
import string
import imageio
import numpy as np
from PIL import Image
from datetime import datetime
from mdutils.tools import Html
from io import BytesIO


def get_datetimestr() -> str:
    time_now = datetime.now()
    time_now_str = time_now.strftime("%Y%m%d_%H%M%S")
    return time_now_str


def get_pid() -> int:
    return os.getpid()


def get_random_string(str_len: int = 10) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=str_len))


def encode_image(image: np.ndarray):
    buffered = BytesIO()
    image = Image.fromarray(image)
    image.save(buffered, format="JPEG")
    output = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return output


def decode_image(base64_string):
    # Step 1: Decode the base64 string
    image_data = base64.b64decode(base64_string)
    # Step 2: Convert to a bytes-like object
    image_bytes = BytesIO(image_data)
    # Step 3: Open the image using PIL
    image = Image.open(image_bytes)
    # Step 4: Convert the image to a numpy array
    image_array = np.array(image)
    return image_array


def get_image_as_message(
    image: Optional[np.ndarray] = None,
    image_path: Optional[str] = None,
    model_name: Optional[str] = None,
    image_detail: str = "low",
):
    # Sanity checks
    assert image is None or image_path is None
    assert not (image is not None and image_path is not None)
    mode = (
        "claude"
        if model_name is not None and model_name.startswith("claude")
        else "openai"
    )

    if image is None:
        image = imageio.imread(image_path, pilmode="RGB")
    image_encoded = encode_image(image)
    img_format = "jpeg"
    if mode == "openai" and model_name.startswith("gpt-"):
        message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{img_format};base64,{image_encoded}",
                "detail": image_detail,
            },
        }
    elif mode == "openai":
        message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/{img_format};base64,{image_encoded}"},
        }
    else:
        message = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{img_format}",
                "data": image_encoded,
            },
        }

    return message


def get_video_as_messages(
    video_path: str,
    model_name: Optional[str] = None,
    subsampling_factor: int = 1,
    image_width: int = None,
    image_detail: str = "low",
):
    messages = []
    with imageio.get_reader(video_path) as reader:
        for i, f in enumerate(reader):
            if i % subsampling_factor != 0:
                continue
            if image_width is not None:
                # Resize to fixed width
                h, w = f.shape[:2]
                image_height = int(float(h) / w * image_width)
                f = cv2.resize(f, (image_width, image_height))
            message = get_image_as_message(
                image=f, model_name=model_name, image_detail=image_detail
            )
            messages.append(message)

    return messages


def convert_content_to_str(
    content: Any, save_dir: str, ignore_images: bool = False
) -> str:
    content_str = ""
    if isinstance(content, str):
        content_str = content + "\n"
    elif isinstance(content, list):
        for c in content:
            if isinstance(c, str):
                content_str += c + "\n"
            elif isinstance(c, dict) and c["type"] == "text":
                content_str += c["text"] + "\n"
            elif isinstance(c, dict) and c["type"] == "image":
                if not ignore_images:
                    # Get file path
                    time.sleep(1)
                    time_now = datetime.now()
                    time_now_str = time_now.strftime("%Y%m%d_%H%M%S")
                    if save_dir is not None:
                        img_save_path = f"{save_dir}/images/image_{time_now_str}.jpg"
                        # Decode base64 string to image
                        img_encoded = c["source"]["data"]
                        img_decoded = decode_image(img_encoded)
                        if img_decoded.shape[2] == 4:
                            mask = img_decoded[..., 3] == 0
                            img_decoded = img_decoded[..., :3]
                            img_decoded[mask, :] = np.array([255, 255, 255])
                        imageio.imwrite(img_save_path, img_decoded)
                        content_str += (
                            "\n\n"
                            + Html.image(
                                path=f"images/image_{time_now_str}.jpg", size="x300"
                            )
                            + "\n\n"
                        )
            elif isinstance(c, dict) and c["type"] == "image_url":
                if not ignore_images:
                    # Get file path
                    time.sleep(1)
                    time_now = datetime.now()
                    time_now_str = time_now.strftime("%Y%m%d_%H%M%S")
                    if save_dir is not None:
                        img_save_path = f"{save_dir}/images/image_{time_now_str}.jpg"
                        # Decode base64 string to image
                        img_encoded = c["image_url"]["url"].split(";base64,")[1]
                        img_decoded = decode_image(img_encoded)
                        if img_decoded.shape[2] == 4:
                            mask = img_decoded[..., 3] == 0
                            img_decoded = img_decoded[..., :3]
                            img_decoded[mask, :] = np.array([255, 255, 255])
                        imageio.imwrite(img_save_path, img_decoded)
                        content_str += (
                            "\n\n"
                            + Html.image(
                                path=f"images/image_{time_now_str}.jpg", size="x300"
                            )
                            + "\n\n"
                        )
                else:
                    content_str += "<image>\n"
            else:
                content_str += "\n" + str(c) + "\n"
    else:
        content_str = str(content)
    return content_str


def count_images_in_query(content: Union[str, list[Any]]):
    n_images = 0
    assert isinstance(content, str) or isinstance(content, list)
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c["type"] in ["image", "image_url"]:
                n_images += 1
    return n_images
