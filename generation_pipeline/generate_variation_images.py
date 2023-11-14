"""
This script generates semi-syntetic variations
by using original images and portrayals created for the original images.
"""

import argparse
import logging
import random
from pathlib import Path

import grounding_dino as gd
import numpy as np
import pandas as pd
import segment_anything as sa
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm

from utils import create_generation_df, find_square

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

def detect(gd_model, image_path, portray):
    """Wrapper function to detect the boxes"""
    boxes, _, _ = gd.predict(
        model=gd_model,
        image_path=image_path,
        prompt=portray
    )
    return boxes

def segment(sam_model, image_source, boxes):
    """Wrapper function to segment an image"""
    # set image
    sam_model.set_image(image_source)

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = gd.box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_model.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    )

    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    image_mask = masks[0][0].cpu().numpy()
    x_s, x_e, y_s, y_e = find_square(image_mask)

    image_source = image_source[y_s:y_e, x_s:x_e, :]
    mask = image_mask[y_s:y_e, x_s:x_e]

    return image_source, mask, (x_s, x_e, y_s, y_e)

def inpaint(stable_diff, prompt, original_image, crop_image_source, crop_mask, bb, **kwargs):
    """Wrapper function to inpaint a image"""
    # resize for inpaint
    image_source = Image.fromarray(crop_image_source).resize((512, 512))
    image_mask = Image.fromarray(crop_mask).resize((512, 512))

    inpaint_im = stable_diff(
        prompt=prompt,
        image=image_source,
        mask_image=image_mask,
        **kwargs
    ).images[0]

    inpaint_im = inpaint_im.resize((bb[1] - bb[0], bb[3] - bb[2]))
    new_image = np.array(original_image)
    new_image[bb[2]:bb[3], bb[0]:bb[1], :] = np.array(inpaint_im)
    new_image = Image.fromarray(new_image)

    return new_image

def generate_variations(gd_model, sam_model, sd_model, image_input, sample, image_folder : Path = None, **kwargs):
    """Generate variations given an original image"""
    save_paths = sample.save
    if image_folder:
        save_paths = image_folder / save_paths
    print(save_paths)
    if all(save_paths.apply(lambda x: Path(x).exists())):
        logging.info('all variations exist')
        return
    item_name = str(sample['item'].values[0])
    item_box = detect(gd_model=gd_model, image_path=image_input, portray=item_name)
    if len(item_box) == 0:
        return
    image_np, _ = gd.load_image(image_input)
    # bb - bounding box (x_l, x_r, y_t, y_b)
    crop_im_src, crop_mask, bb = segment(sam_model=sam_model, image_source=image_np, boxes=item_box)

    for idx, exp in sample.iterrows():
        # pbar.set_postfix(var=exp.short, **_attr)
        try:
            new_image = inpaint(
                stable_diff=sd_model,
                prompt=exp.desc,
                original_image=image_np,
                crop_image_source=crop_im_src,
                crop_mask=crop_mask,
                bb = bb,
                **kwargs
            )
            print(str(save_paths.loc[idx]))
            new_image.save(str(save_paths.loc[idx]))
        except Exception as err:
            print(err)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default='portrayals.json')
    parser.add_argument("--gd-config", default="../model_lib/groundingdino_swint_ogc.py")
    parser.add_argument("--gd-checkpoint", default="../model_lib/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam-checkpoint", default='../model_lib/sam_vit_h_4b8939.pth')
    parser.add_argument("--sd-checkpoint", default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument("--device", type=torch.device, default='cuda')
    parser.add_argument("--output-dir", type=Path, default='../outputs')
    parser.add_argument("--input-folder", type=Path, default='../assets')
    args = parser.parse_args()

    df = pd.read_json(args.input_file)
    gen_df = create_generation_df(
        df=df,
        save_path=args.output_dir / 'variations.json'
    )

    _gd = gd.load_model(
        config_path=args.gd_config,
        model_path=args.gd_checkpoint,
    )
    _sam = sa.SamPredictor(sa.build_sam(checkpoint=args.sam_checkpoint))
    _sd =  StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_checkpoint,
        torch_dtype=torch.float16,
    )
    _sd.set_progress_bar_config(leave=False)
    _sd = _sd.to(args.device)

    groups = [(k, _df) for k, _df in gen_df.groupby(['origin_name', 'source', 'item'])]
    random.shuffle(groups)
    pbar = tqdm(groups)

    for (origin_name, source, item), exps in pbar:
        _attr = {'file': origin_name, 'item' : item}
        pbar.set_postfix(_attr)
        output_im_dir = args.output_dir / 'variations' / origin_name / item
        output_im_dir.mkdir(parents=True, exist_ok=True)
        im_source = args.input_folder / source
        generate_variations(
            _gd, _sam, _sd, im_source, exps,
            image_folder = args.output_dir / 'variations'
        )
