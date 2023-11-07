"""
This file evaluates the semi-syntetic image variations test dataset
"""
import json
from pathlib import Path
from typing import List
from collections import OrderedDict

import clip
import numpy as np
import torch

from PIL import Image
from tqdm import tqdm

import datasets

_model, _preprocess = None, None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def summarize(result_list : List):
    """Calculate overall accuracy and compiles human readable evaluation results.

    Args:
        result_list (List): The list of results for all dataset.

    Returns:
        tuple(str, dict): Returns human readable output and scores.
    """
    con_fn = lambda res, x : 100 * np.concatenate([r[x] for r in res]).mean()
    mean_fn = lambda res, x: 100 * np.mean([r[x] for r in res])

    scores = OrderedDict({
        'Text 1': con_fn(result_list, 0),
        'Image 1': con_fn(result_list, 1),
        'Group 1': con_fn(result_list, 2),
        'Text All': mean_fn(result_list, 3),
        'Image All': mean_fn(result_list, 4),
        'Group All': mean_fn(result_list, 5)
    })

    return_string = '\n'.join(
        f'{k}\t: %{v:.4}' for k, v in scores.items()
    )

    print(return_string)
    return return_string, scores

def process_sample(row : tuple):
    """Process one sample from dataset. This function
        1) loads the captions
        2) opens and preprocesses the images
        3) tokenizes captions
        4) feeds the images and tokenized captions into model
        5) creates scores

    Args:
        items (List): The list of items in the sample. Each item consist of a image url and caption
        image_root (str, optional): _description_. Defaults to None.

    Returns:
        tuple[List, ]: _description_
    """
    row = row[1]
    texts = [
        row[f'caption_{c}'] for c in range(row.num_sample)
    ]
    images = [
        _preprocess(Image.open(row[f'image_{c}']['path']).convert("RGB"))
        for c in range(row.num_sample)
    ]

    images = torch.stack(images, axis=0)

    with torch.no_grad():
        tokenized = clip.tokenize(texts)
        logits = _model(images.to(_DEVICE), tokenized.to(_DEVICE))[0].cpu().numpy()

    grount_truth = np.arange(row.num_sample)

    text_1 = logits.argmax(1) == grount_truth
    image_1 = logits.argmax(0) == grount_truth
    group_1 = np.logical_and(text_1, image_1)
    text_all = np.all(text_1)
    image_all = np.all(image_1)
    group_all = np.all(group_1)

    return (text_1, image_1, group_1, text_all, image_all, group_all)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate Negative Mining validation dataset evaluations')
    parser.add_argument('--model-name', default=None, required=True, type=str,
        help='The tag of the model')
    parser.add_argument('--snapshot-file', default=None, required=False, type=Path,
        help='The model to evaluate')
    parser.add_argument('--snapshot-folder', default=None, required=False, type=Path,
        help='The folder to search experiment results. If snapshot_file is specified, this is neglected.')
    parser.add_argument('--save-path',  default='output/gnm_results/', type=Path,
        help='The path to store evaluation files')
    parser.add_argument('--hf-auth-token', type=str,
        help='The path to store evaluation files')
    parser.add_argument('--force',  action='store_true',
        help='Specify to overwrite existing evaluation of a model')
    parser.add_argument('--clip-model', default='ViT-B/32', type=str,
        help='The selection of the CLIP model.')
    args = parser.parse_args()

    assert args.snapshot_file or args.snapshot_folder, 'You should either set snapshot file or folder'
    # If snapshot file is specified just load the file
    # Otherwise search for .pt files in snapshots folder and evaluate them

    if args.snapshot_file:
        snapshots = [str(args.snapshot_file)]
    else:
        snapshots = sorted(
            list(args.snapshot_folder.glob('*.pt')),
            key =lambda x:int(x.name.rsplit('_', 1)[-1])
        )
        print(f'There are #{len(snapshots)} snapshots')

    dataset = datasets.load_dataset(
        'ugursahin/generative-negative-mining-dataset',
        use_auth_token=args.hf_auth_token
    )
    dataset = dataset['test'].to_pandas()

    # Load Models & Tokenizers
    _model, _preprocess = clip.load(args.clip_model, device=_DEVICE)
    _model = _model.eval()

    args.save_path.mkdir(parents=True, exist_ok=True)
    # The paths to save the results
    json_save_path = args.save_path / f"{args.model_name}.json"
    txt_save_path = args.save_path / f"{args.model_name}.txt"

    # Check if force is specified
    OPEN_MODE = ['a', 'w'][bool(args.force)]

    # If the model is evaluated before, fetch the results
    if json_save_path.exists() and not args.force:
        print(f'evaluation file has found: {json_save_path}')
        with json_save_path.open() as fd:
            result_dict = json.load(fd)
    else:
        result_dict = {}

    # For each of the snapshot, calculate the
    # scores and save the results into a summary string
    # and dictionary.

    RESULT_TEXT = ''
    for snap in snapshots:
        if str(snap) in result_dict:
            continue
        RESULT_TEXT += f'{snap}\n'
        print(snap)
        _model.load_state_dict(torch.load(snap)['model_state_dict'])

        snap_scores = list(
            map(
                process_sample,
                tqdm(dataset.iterrows())
            )
        )

        score_txt, score_dict = summarize(snap_scores)
        RESULT_TEXT += score_txt
        result_dict[str(snap)] = score_dict

    # Save the dictionary as string
    with open(json_save_path, 'w', encoding='utf-8') as fd:
        json.dump(result_dict, fd)
    with open(txt_save_path, OPEN_MODE, encoding='utf-8') as fd:
        fd.write(RESULT_TEXT)
