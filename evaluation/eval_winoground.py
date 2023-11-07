"""
This script evaluates the winoground datatet.
"""
import json
from pathlib import Path
from PIL import Image

import clip
import torch
from datasets import load_dataset
from winoground_nums import summarize

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load Models & Tokenizers
clip_model, clip_preprocess = None, None

def process_sample(row):
    """Process one sample from dataset. This function
        1) opens two images of the sample and preprocess
        3) tokenizes captions
        4) feeds the images and tokenized captions into model
        5) calculates text, image and group score
    """

    print(f"\r {row.name}  ", end="")
    image_0, image_1 = row['image_0']['path'], row['image_1']['path']

    image_0 = clip_preprocess(Image.open(image_0).convert("RGB")).to(_DEVICE)
    image_1 = clip_preprocess(Image.open(image_1).convert("RGB")).to(_DEVICE)
    images = torch.stack([image_0, image_1], axis=0)
    captions = row[['caption_0', 'caption_1']].values

    with torch.no_grad():
        tokenized = clip.tokenize(captions)
        probs = clip_model(images.to(_DEVICE), tokenized.to(_DEVICE))[0].cpu().numpy()
        text_score = probs[0][0] > probs[0][1] and probs[1][1] > probs[1][0]
        image_score = probs[0][0] > probs[1][0] and probs[1][1] > probs[0][1]
        group_score = text_score and image_score

    return [text_score, image_score, group_score, probs[0][0], probs[1][0], probs[0][1], probs[1][1]]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='winoground evaluation')
    parser.add_argument('--model_name', default=None, required=True,
        help='The tag of the model for saving purposes')
    parser.add_argument('--snapshot_file', default=None, required=False, type=Path,
        help='The model to evaluate')
    parser.add_argument('--snapshot_folder', default=None, required=False, type=Path,
        help='The folder to search experiment results. If snapshot_file is specified, this is neglected.')
    parser.add_argument('--save_path',  default='autogmentation_results', type=Path,
        help='The path to store evaluation files')
    parser.add_argument('--force',  action='store_true',
        help='Specify to overwrite existing evaluation of a model')
    parser.add_argument('--huggingface_auth_key',  type=str, default=None, required=True,
        help='In order to download winoground, you need to have a huggingface auth key')
    args = parser.parse_args()

    assert args.snapshot_file or args.snapshot_folder, 'You should either set snapshot file or folder'
    
    # Load Models & Tokenizers
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=_DEVICE)
    clip_model = clip_model.eval()
    
    # The paths to save the results
    json_save_path = args.save_path / f"json/{args.model_name}.json"
    txt_save_path = args.save_path / f"txt/{args.model_name}.txt"

    OPEN_MODE = ['a', 'w'][bool(args.force)]

    if json_save_path.exists() and not args.force:
        print(f'evaluation file has found: {json_save_path}')
        with json_save_path.open() as fd:
            result_dict = json.load(fd)
    else:
        result_dict = {}

    examples = load_dataset('facebook/winoground', use_auth_token=args.huggingface_auth_key)
    df = examples['test'].to_pandas()

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

    RESULT_TEXT = ''
    for snap in snapshots:
        if str(snap) in result_dict:
            continue
        RESULT_TEXT += f'{snap}\n'
        print(snap)
        clip_model.load_state_dict(torch.load(snap)['model_state_dict'])
        snap_results = df.apply(process_sample, axis=1, result_type='expand')
        score_txt, score_dict = summarize(snap_results)
        RESULT_TEXT += score_txt
        result_dict[str(snap)] = score_dict

    # Save the dictionary as string
    with open(txt_save_path, OPEN_MODE, encoding='utf-8') as fd:
        fd.write(RESULT_TEXT)
    with open(json_save_path, 'w', encoding='utf-8') as fd:
        json.dump(result_dict, fd)
