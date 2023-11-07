"""
This script is the wrapper for grounding dino.
Please refer to https://github.com/IDEA-Research/GroundingDINO for the project.
"""

import logging

import numpy as np
import torch
from groundingdino.util.inference import load_model as _load_model, load_image, predict as _predict
from groundingdino.util import box_ops
from PIL import Image

def load_model(
        config_path = "../model_lib/groundingdino_swint_ogc.py",
        model_path = "../model_lib/groundingdino_swint_ogc.pth"
    ):
    return _load_model(config_path, model_path)

def predict(
    model,
    image_path : str = None,
    image : Image = None,
    box_threshold : float = 0.35,
    text_threshold : float = 0.25,
    prompt : str = None,
    device : torch.device = torch.device('cuda')
):

    if image_path:
        _, image = load_image(image_path)

    elif not torch.is_tensor(image):
        logging.error('Image parameter should be a tensor')
        return

    boxes, logits, phrases = _predict(
        model=model,
        image=image.to(device),
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return boxes, logits, np.array(phrases)
