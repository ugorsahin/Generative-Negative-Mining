"""
Tag2Text Inference Module

This module consists of functions to perform Tag2Text inference on a collection of images
based on the specified command-line arguments. It utilizes Tag2Text and Grounding DINO
models to extract information about objects, generate captions, and optionally
save annotated images. The processed results are then saved in a JSON file.

Please refer to https://github.com/xinyu1205/recognize-anything for the research repository.
"""


import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torchvision

from PIL import Image
from ram import get_transform
from ram import inference_tag2text as inference
from ram.models import tag2text
from tqdm import tqdm

import grounding_dino as gd
from utils import prepare_image_folder

def load_model(
    pretrained  : str,
    image_size  : float = 384,
    threshold   : float = 0.64,
    delete_tag_index : List[int] = None,
    vit : str = 'swin_b',
    device : str = None):
    """
    Load a pre-trained Tag2Text model along with its associated image transformation.

    Parameters:
        - pretrained (str): Path to a tag2text checkpoint.
        - image_size (int, optional): Size of the input images for the model (default is 384).
        - threshold (float, optional): Threshold value for tagging confidence (default is 0.64).
        - delete_tag_index (list, optional): List of tag indices to be deleted from the default set
        - vit (str, optional): Vision Transformer (ViT) variant to use (default is 'swin_b').
        - device (str, optional): Device to which the model should be moved

    Returns:
        Tuple[tag2text_model, tag2text_transform]:
            - tag2text_model: Loaded Tag2Text model with specified configurations.
            - tag2text_transform: Image transformation function used by the model.
    """

    # Delete some tags that may disturb captioning
    # 127: "quarter"; 2961: "back", 3351: "two"; 3265:"three";
    # 3338: "four"; 3355: "five"; 3359: "one"

    if delete_tag_index is None:
        delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]
    # Load model
    tag2text_model = tag2text(
        pretrained=pretrained,
        image_size=image_size,
        vit=vit,
        delete_tag_index=delete_tag_index
    )
    # set threshold for tagging
    tag2text_model.threshold = threshold
    tag2text_model.eval()

    # Move the model to the device if provided
    if device:
        tag2text_model = tag2text_model.to(device)

    # Get the image transformation function used by the model
    tag2text_transform = get_transform(image_size=image_size)

    return tag2text_model, tag2text_transform

def get_singular_phrases(phrase_list : List[str]) -> List[str]:
    """
    Filter a list of phrases to include only those that appear just once.

    This function takes a list of phrases and returns a new list
    containing only the phrases that appear exactly once in the input list.

    Parameters:
        - phrase_list (List[str]): A list of phrases.

    Returns:
        - List[str]: A list containing singular phrases.
    """
    # Count the occurrences of each phrase in the input list
    object_sizes = Counter(phrase_list)

    # Filter phrases to include only those that appear exactly once
    single_objects = list(map(
        lambda phrase: object_sizes[phrase] == 1,
        object_sizes
    ))
    return single_objects

def extract_tags(
        tag2text_model,
        gd_model,
        tag2text_transform,
        image_path          : Path,
        iou_threshold       : float = 0.25,
        singular_objects    : bool  = False,
        save_annotation     : str   = None,
        device : Union[str, torch.device] = None,
    ):
    """
    Process a single image using Tag2Text and Grounding DINO models,
    extracting information about objects and generating annotations
    based on the provided parameters.

    Parameters:
        - tag2text_model: The Tag2Text model for object tagging.
        - gd_model: The Grounding DINO model for object grounding.
        - tag2text_transform: Image transformation function used by the Tag2Text model.
        - image_path (Path): Path to the input image.
        - iou_threshold (float): Intersection over Union (IoU) threshold for NMS (default is 0.25).
        - singular_objects (bool): If True, only detects singular objects (default is False).
        - save_annotation (str): If set, save the annotated image to given path (default is None).
        - device (str, torch.device): Device on which to perform inference (default is None).

    Returns:
        Tuple[box_tags, caption, boxes]:
            - box_tags: List of grounded object tags.
            - caption: Generated caption describing the image.
            - boxes: List of bounding boxes corresponding to the grounded objects.
    """
    # Load and transform the input image
    raw_image = Image.open(image_path)
    image = tag2text_transform(raw_image).unsqueeze(0).to(device)

    # Send the image to Tag2Text model for object tagging
    res = inference(image, tag2text_model)
    objects, caption = res[0].replace(' |', ','), res[2]

    logging.info("Caption: %s", caption)
    logging.info("Tags: %s", str(objects))

    # Run Grounding DINO model for object grounding
    boxes, scores, box_tags = gd.predict(
        model=gd_model, image_path=image_path, prompt=objects
    )

    # Filter out singular phrases if specified
    if singular_objects:
        s_map = get_singular_phrases(box_tags)
        boxes, scores, box_tags = boxes[s_map], scores[s_map], box_tags[s_map]
    boxes = boxes.cpu()

    # Convert bounding boxes to image coordinates
    H, W = raw_image.size
    for i in range(boxes.shape[0]):
        boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
        boxes[i][:2] -= boxes[i][2:] / 2
        boxes[i][2:] += boxes[i][:2]

    logging.info("Before NMS: %d boxes", box_tags.shape[0])
    # Apply non-maximum suppression
    nms_idx = torchvision.ops.nms(boxes, scores, iou_threshold).numpy().tolist()
    box_tags = box_tags[nms_idx]
    boxes = boxes[nms_idx]
    logging.info("After NMS: %d boxes", box_tags.shape[0])

    # Save annotated image if specified
    if save_annotation:
        annotated_frame = gd.annotate(
            image_source=np.array(raw_image), boxes=boxes, logits=scores, phrases=box_tags)
        Image.fromarray(annotated_frame).save(save_annotation)
    return box_tags, caption, boxes

def main(args : argparse.Namespace):
    """
    Perform Tag2Text inference on images based on the specified command-line arguments.

    Parameters:
        - args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    # Set logging level to INFO if verbose mode is enabled
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Create a folder for saving annotated images if specified
    if args.save_annotated_image:
        ann_path = args.save_folder / 'annotations'
        ann_path.mkdir(parents=True, exist_ok=True)

    # Load image data based on the source (image folder or annotation file)
    if args.image_folder:
        image_data = prepare_image_folder(args.root_folder, args.image_folder)

    elif args.annotation_file:
        with (args.root_folder / args.annotation_file).open() as fd:
            image_data = json.load(fd)

    model_tag2text, transform = load_model(
        pretrained=str(args.root_folder / args.pretrained),
        image_size=args.image_size,
        threshold=args.threshold,
        vit = 'swin_b',
        device=args.device
    )
    model_gd = gd.load_model()

    # Process each image in the dataset
    image_tag_dict = []
    for item in tqdm(image_data):
        # Set the path for saving annotated images if specified
        save_ann_path = ann_path / Path(item['image']).name if args.save_annotated_image else None
        # Construct the full path to the image
        image_path = args.root_folder / item['image']

        # Extract tags and Tag2Text caption for the current image
        tags, tag2text_caption, _ = extract_tags(
            tag2text_model      = model_tag2text,
            gd_model            = model_gd,
            tag2text_transform  = transform,
            image_path          = image_path,
            device              = args.device,
            save_annotation     = save_ann_path
        )

        # Append information about the processed image as a record
        image_tag_dict.append({
            'save_path'         : image_path.name,
            'image'             : item['image'],
            'original_captions' : item['original_captions'],
            'tag2text_caption'  : tag2text_caption,
            'items'             : tags
        })

    # Save the results to a JSON file
    with (args.save_folder / 'tag2text.json').open('w') as fd:
        json.dump(image_tag_dict, fd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tag2Text inferece for tagging and captioning')
    parser.add_argument('--image-folder', default='assets/', type=Path,
                        help='path to image folder')
    parser.add_argument('--root-folder', default='../', type=Path,
                        help='path to image folder')
    parser.add_argument('--annotation-file', default='assets/coco_karpathy_train.json',
                        help='path to dataset',)
    parser.add_argument('--save-folder',default='../output/', type=Path,
                        help='path to save annotations')
    parser.add_argument('--save-annotated-image', action='store_true',
                        help='If true, saves the annotated image')
    parser.add_argument('--force', action='store_true',
                        help='If true, force saves')
    parser.add_argument('--pretrained', default='model_lib/tag2text_swin_14m.pth',
                        help='path to pretrained model')
    parser.add_argument('--image-size', default=384, type=int,
                        help='input image size (default: 448)')
    parser.add_argument('--threshold', default=0.64, type=float,
                        help='threshold value')
    parser.add_argument('--verbose', action='store_true',
                        help='If true, logging info redirected to stdin')
    parser.add_argument('--device', default=None,
                        help='Device to perform actions')
    _args = parser.parse_args()
    _args.device = torch.device(_args.device)
    _args.save_folder.mkdir(parents=True, exist_ok=_args.force)

    main(_args)
