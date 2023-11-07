"""
This script calculate two scores used to filter generations, itm_score and iva_score.
For ITM Score, it uses BLIP model
Please refer https://github.com/salesforce/BLIP to learn more about BLIP
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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

def process_image(model, vis_pre, text_pre, captions, device='cuda', image=None, image_path=None):
    """Process and image and return itm scores for given captions"""
    if image_path:
        image = Image.open(image_path).convert("RGB")

    img = vis_pre["eval"](image).unsqueeze(0).to(device)
    txts = [text_pre["eval"](cap) for cap in captions]

    with torch.no_grad():
        blip_logits = [
            model({"image": img, "text_input": cap}, match_head="itm").cpu().numpy()[0, 1]
            for cap in txts
        ]
    return blip_logits

def image_variation_area(image_paths, std_mean_threshold=10):
    """
    Calculate standard deviation of batch per pixel per channel
    (B, W, H, C) -> (W, H, C)
    """
    images = np.stack(
        list(map(
            lambda image_path: np.asarray(
                Image.open(image_path),
                dtype=np.uint8
            ),
            image_paths
        ))
    ).std(0)
    # Calculate the mean of standard deviation in channel dimension
    # (W, H, C) - > (W, H)
    immask = images.mean(2) > std_mean_threshold
    # Calculate and return average number of pixels with high average standard deviance
    return 100 * immask.mean()

def main(args):
    """Main function to run this script"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blip_model, blip_vis, blip_txt = load_blip_model(args.blip_model, device)

    data = pd.read_json(args.input_file).to_dict('records')
    iva_cache = {}
    processed = {}
    if os.path.exists(args.output_file):
        processed = pd.read_json(args.output_file, orient='records').to_dict('records')
        processed = {
            i['name'] : i for i in processed
        }

    for item in tqdm(data):
        if item.get('v_score'):
            continue

        im_key = item['name']
        if processed.get(im_key):
            continue

        out = process_image(
            model=blip_model,
            vis_pre=blip_vis,
            text_pre=blip_txt,
            image_path=args.image_folder / item['source'],
            captions=item['captions'],
            device=device
        )
        item['v_score'] = out
        iva_score = iva_cache.get(im_key)
        if not iva_score:
            iva_score = image_variation_area(
                (args.image_folder / item['origin_name'] / item['item']).glob('*.jpg')
            )
            item['iva_score'] = iva_score
            iva_cache[im_key] = iva_score

        processed[im_key] = item

    pd.DataFrame(list(processed.values())).to_json(args.output_file, orient='records')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--blip-model", default='blip_image_text_matching')
    parser.add_argument("--input-file", default='../outputs/variation_dataset.json')
    parser.add_argument("--output-file", default='../outputs/variation_scored.json')
    parser.add_argument("--image-folder", default='../outputs/variations', type=Path)

    args = parser.parse_args()

    main(args)
