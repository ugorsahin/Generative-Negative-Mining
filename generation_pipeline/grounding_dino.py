"""
This script is the wrapper for grounding dino.
Please refer to https://github.com/IDEA-Research/GroundingDINO for the project.
"""

import logging
from typing import Union

import numpy as np
import torch
from groundingdino.util.inference import (
    load_model as _load_model,
    predict as _predict,
    load_image,
    annotate
)
from groundingdino.util import box_ops
from PIL import Image

def load_model(
    config_path : str = "../model_lib/groundingdino_swint_ogc.py",
    model_path : str = "../model_lib/groundingdino_swint_ogc.pth"):
    """
    Load the Grounding DINO model.

    Parameters:
        - config_path (str): Path to the model configuration file.
        - model_path (str): Path to the model checkpoint file.

    Returns:
        torch.nn.Module: Loaded Grounding DINO model.
    """
    return _load_model(config_path, model_path)

def predict(
    model           : torch.nn.Module,
    image_path      : str = None,
    image           : Image = None,
    box_threshold   : float = 0.35,
    text_threshold  : float = 0.25,
    prompt          : str = None,
    device          : Union[torch.device, str] = None
):
    """
    Perform prediction using the Grounding DINO model.

    Parameters:
        - model (torch.nn.Module): Loaded Grounding DINO model.
        - image_path (str): Path to the input image file.
        - image (Image): PIL Image object.
        - box_threshold (float): Threshold for bounding box confidence.
        - text_threshold (float): Threshold for text confidence.
        - prompt (str): Caption prompt for grounding.
        - device (torch.device): Device for model inference.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Predicted boxes, logits, and phrases.
    """

    # Check if an image path is provided
    if image_path:
        logging.info('Loading image %s', image_path)
        # Load the image using the provided image path
        _, image = load_image(image_path)

    # If no image path is provided, check if the image parameter is a tensor
    # Raise error otherwise
    assert torch.is_tensor(image), 'Image parameter should be a tensor'

    # If a device is specified, move the image to that device
    if device:
        image = image.to(device)

    # Perform prediction using the Grounding DINO model
    boxes, logits, phrases = _predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    logging.info('Total number of %d boxes found', len(boxes))
    return boxes, logits, np.array(phrases)
