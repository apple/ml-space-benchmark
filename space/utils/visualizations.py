# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import cv2
import numpy as np


def add_goal_to_obs(obs_img: np.ndarray, goal_img: np.ndarray) -> np.ndarray:
    H, W, _ = obs_img.shape

    goal_img = np.copy(goal_img)
    goal_img = cv2.resize(goal_img, (W // 6, H // 6))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (5, 25)

    # fontScale
    fontScale = 0.7

    color = (0, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    goal_img = cv2.putText(
        goal_img, "Goal", org, font, fontScale, color, thickness, cv2.LINE_AA
    )

    # Draw border
    goal_img = cv2.rectangle(
        goal_img, (0, 0), (goal_img.shape[1] - 1, goal_img.shape[0] - 1), color, 2
    )

    obs_img = np.copy(obs_img)
    obs_img[5 : 5 + goal_img.shape[0], W - 5 - goal_img.shape[1] : W - 5] = goal_img

    return obs_img


def add_text_to_image(
    img: np.ndarray,
    text: str,
    origin: list[int] = None,
    add_background: bool = False,
    font_scale: int = 1,
    color=(255, 255, 255),
    thickness=2,
):
    img = np.copy(img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Using cv2.putText() method
    textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
    if origin is None:
        org = ((img.shape[1] - textsize[0]) // 2, textsize[1] + 20)
    else:
        org = tuple(origin)
    # Add black background if needed
    if add_background:
        start_x = org[0] - 5
        end_x = start_x + textsize[0] + 10
        start_y = org[1] - textsize[1] - 5
        end_y = start_y + textsize[1] + 10
        img_crop = img[start_y:end_y, start_x:end_x]
        img_blend = cv2.addWeighted(img_crop, 0.3, np.zeros_like(img_crop), 0.7, 1.0)
        img[start_y:end_y, start_x:end_x] = img_blend

    img = cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    return img
