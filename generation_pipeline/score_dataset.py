"""
This script calculate two scores used to filter generations, itm_score and iva_score.
For ITM Score, it uses BLIP model
Please refer https://github.com/salesforce/BLIP to learn more about BLIP
"""

import argparse
import json
from typing import Any, List, Dict, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

def load_blip_model(model_name = 'blip_image_text_matching', device = 'cuda'):
    """Loads BLIP model from remote server"""
    model, vis_pre, text_pre = load_model_and_preprocess(
        model_name,
        "large",
        device=device,
        is_eval=True
    )
    return model, vis_pre, text_pre

def process_image(
    model       : torch.nn.Module ,
    vis_pre     : Dict[str, Any],
    text_pre    : Dict[str, Any],
    captions    : List[str],
    device      : Union[str, torch.device] = 'cuda',
    image       : Image.Image = None,
    image_path  : str = None):
    """
    Process an image using the BLIP model by combining visual and textual information.

    Parameters:
        - model (torch.nn.Module): BLIP model.
        - vis_pre (Dict[str, Any]): Dictionary containing visual preprocessor functions.
        - text_pre (Dict[str, Any]): Dictionary containing text preprocessor functions.
        - captions (List[str]): List of textual captions.
        - device (Union[str, torch.device]): Device for model inference.
        - image (PIL.Image.Image): PIL Image object representing the input image.
        - image_path (str): Path to the input image file.

    Returns:
        List[float]: List of BLIP logits corresponding to each input caption.
    """
    if image_path:
        image = Image.open(image_path).convert("RGB")

    # Preprocess the image using the visual preprocessor of BLIP
    img = vis_pre["eval"](image).unsqueeze(0).to(device)
    # Preprocess the captions using the text preprocessor of BLIP.
    txts = list(map(
        text_pre["eval"],
        captions
    ))

    # Perform BLIP model inference for each caption
    with torch.no_grad():
        # Calculate BLIP logits for each caption and convert to a list
        blip_logits = list(map(
            lambda x: model({"image": img, "text_input": x}, match_head="itm").cpu().numpy()[0, 1],
            txts
        ))
    return blip_logits

def image_variation_area(image_paths : List[str], std_mean_threshold : int = 10):
    """
    Calculate the image variation area based on standard deviation per pixel per channel.

    Parameters:
        - image_paths (List[str]): List of paths to image files.
        - std_mean_threshold (int): Threshold for the mean standard deviation. Default is 10.

    Returns:
        float: Average percentage of pixels with high average standard deviation.

    Notes:
        The function loads a batch of images specified by the provided paths,
        calculates the standard deviation per pixel per channel, and then computes
        the average percentage of pixels with a high average standard deviation.

    Example:
        >>> paths = ['image1.jpg', 'image2.jpg']
        >>> result = image_variation_area(paths, std_mean_threshold=12)
        >>> print(result)
        18.5
    """

    # Load images and calculate standard deviation
    images = np.stack(
        list(map(
            lambda image_path: np.asarray(Image.open(image_path), dtype=np.uint8),
            image_paths
        ))
    ).std(0)

    # Calculate the mean of standard deviation in channel dimension
    # (W, H, C) - > (W, H)
    immask = images.mean(2) > std_mean_threshold

    # Calculate and return average number of pixels with high average standard deviance
    return 100 * immask.mean()

def main(args : argparse.Namespace):
    """
    Main function to run the script.

    Parameters:
        - args (argparse.Namespace): Command-line arguments obtained from argparse.ArgumentParser.

    Returns:
        None

    Notes:
        This function serves as the main entry point for the script. It loads the BLIP model,
        processes input data, calculates image scores using BLIP model and image variation area,
        and writes the processed data to an output file.
    """

    # Load BLIP model and preprocessors
    blip_model, blip_vis, blip_txt = load_blip_model(args.blip_model, args.device)

    # Load input data from the specified file
    with args.input_file.open() as fd:
        data = json.load(fd)

    # Initialize caches for image variation area and processed data
    iva_cache = {}
    processed = {}

    # Read previously processed data from the output file, if it exists
    if args.output_file.exists():
        with args.output_file.exists() as fd:
            processed = json.load(fd)
            processed = {i['name'] : i for i in processed}

    # Process each item in the input data
    for item in tqdm(data):
        # Skip items that already have a 'itm_score' field
        if item.get('itm_score'):
            continue

        # Skip items that have already been processed
        im_key = item['name']
        if processed.get(im_key):
            continue

        # Process the image using BLIP model
        item['itm_score'] = process_image(
            model=blip_model,
            vis_pre=blip_vis,
            text_pre=blip_txt,
            image_path=args.image_folder / item['source'],
            captions=item['captions'],
            device=args.device
        )

        # Calculate and cache the image variation area score
        iva_score = iva_cache.get(im_key)
        if not iva_score:
            iva_score = image_variation_area(
                (args.image_folder / item['origin_name'] / item['item']).glob('*.jpg')
            )
            item['iva_score'] = iva_score
            iva_cache[im_key] = iva_score
        # Update the processed data
        processed[im_key] = item

    with args.output_file.open('w') as fd:
        json.dump(list(processed.values()), fd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--blip-model", type=str, default='blip_image_text_matching')
    parser.add_argument("--input-file", type=Path, default='../outputs/variation_dataset.json')
    parser.add_argument("--output-file", type=Path, default='../outputs/variation_scored.json')
    parser.add_argument("--image-folder", type=Path, default='../outputs/variations')
    parser.add_argument('--device', type=torch.device, default='cuda')
    arguments = parser.parse_args()
    main(arguments)
