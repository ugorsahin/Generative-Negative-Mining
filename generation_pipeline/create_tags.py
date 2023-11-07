import argparse
import json
import logging

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from PIL import Image
from ram import get_transform
from ram import inference_tag2text as inference
from ram.models import tag2text
from tqdm import tqdm

import grounding_dino as gd
from utils import prepare_image_folder

# Delete some tags that may disturb captioning
# 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
def load_model(
        pretrained=None,
        image_size=384,
        threshold=0.64,
        delete_tag_index = None,
        vit = 'swin_b',
        device=None
    ):
    if delete_tag_index is None:
        delete_tag_index = [127,2961, 3351, 3265, 3338, 3355, 3359]
    # Load model
    tag2text_model = tag2text(
        pretrained=pretrained,
        image_size=image_size,
        vit=vit,
        delete_tag_index=delete_tag_index
    )
    tag2text_model.threshold = threshold  # threshold for tagging
    tag2text_model.eval()

    if device:
        tag2text_model = tag2text_model.to(device)

    tag2text_transform = get_transform(image_size=image_size)

    return tag2text_model, tag2text_transform

def filter_multiple_instances(phrase_list):
    object_sizes = Counter(phrase_list)
    single_objects = [k for k, v in object_sizes.items() if len(v) == 1]
    return single_objects

def process_one_image(
        tag2text_model,
        gd_model,
        tag2text_transform,
        image_path : str,
        iou_threshold : float =0.25,
        singular_objects : bool =False,
        device : torch.device =None,
        save_annotation : str = None,
    ):

    raw_image = Image.open(image_path)
    image = tag2text_transform(raw_image).unsqueeze(0).to(device)

    res = inference(image, tag2text_model)
    objects, caption = res[0].replace(' |', ','), res[2]

    logging.info("Caption: %s", caption)
    logging.info("Tags: %s", str(objects))

    # run grounding dino model
    boxes, scores, box_tags = gd.predict(
        model=gd_model, image_path=image_path, prompt=objects
    )

    if singular_objects:
        s_map = filter_multiple_instances(box_tags)
        boxes, scores, box_tags = boxes[s_map], scores[s_map], box_tags[s_map]
    boxes = boxes.cpu()

    H, W = raw_image.size
    for i in range(boxes.shape[0]):
        boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
        boxes[i][:2] -= boxes[i][2:] / 2
        boxes[i][2:] += boxes[i][:2]

    logging.info("Before NMS: %d boxes", box_tags.shape[0])
    nms_idx = torchvision.ops.nms(boxes, scores, iou_threshold).numpy().tolist()
    box_tags = box_tags[nms_idx]
    boxes = boxes[nms_idx]
    logging.info("After NMS: %d boxes", box_tags.shape[0])

    if save_annotation:
        annotated_frame = gd.annotate(
            image_source=np.array(raw_image), boxes=boxes, logits=scores, phrases=box_tags)
        Image.fromarray(annotated_frame).save(save_annotation)
    return box_tags, caption, boxes

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
                        help='If true, talks')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.save_folder.mkdir(exist_ok=args.force)

    model_tag2text, transform = load_model(
        pretrained=str(args.root_folder / args.pretrained),
        image_size=args.image_size,
        threshold=args.threshold,
        vit = 'swin_b',
        device=_device
    )
    model_gd = gd.load_model()

    if args.image_folder:
        image_data = prepare_image_folder(args.root_folder, args.image_folder)

    elif args.annotation_file:
        with (args.root_folder / args.annotation_file).open() as fd:
            image_data = json.load(fd)

    image_tag_dict = []

    if args.save_annotated_image:
        ann_path = args.save_folder / 'annotations'
        ann_path.mkdir(exist_ok=True)

    for item in tqdm(image_data):
        save_ann_path = ann_path / Path(item['image']).name if args.save_annotated_image else None
        image_path = str(args.root_folder / item['image'])
        tags, tag2text_caption, _ = process_one_image(
            tag2text_model      = model_tag2text,
            gd_model            = model_gd,
            tag2text_transform  = transform,
            image_path          = image_path,
            device              =_device,
            save_annotation     = save_ann_path
        )

        image_tag_dict.append({
            'save_path'         : image_path.name,
            'image'             : item['image'],
            'original_captions' : item['original_captions'],
            'tag2text_caption'  : tag2text_caption,
            'items'             : tags
        })

    pd.DataFrame(image_tag_dict).to_json(args.save_folder / 'tag2text.json', orient='records')
