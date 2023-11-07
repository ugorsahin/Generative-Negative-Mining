"""
This script is the wrapper for Stanford Stanza.
It contains utility functions.
Please refer to https://stanfordnlp.github.io/stanza/ to learn more about Stanza
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import stanza
from tqdm import tqdm

STANZA_MODEL = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def tree2text(tree):
    "Converts a constituency tree into a string"
    if len(tree.children) == 0:
        return str(tree) + " "
    return ''.join([tree2text(child) for child in tree.children])

def extract_nouns(tree):
    """Extracts nouns from a constituency tree"""
    level_down = any(k.label in ('S', 'NP', 'VP', 'PP') for k in tree.children)

    values = []
    if level_down:
        values.extend([extract_nouns(child) for child in tree.children])
    else:
        values.append(tree2text(tree))

    nouns = []
    for i in values:
        if isinstance(i, list):
            nouns.extend(i)
        else:
            nouns.append(i)
    return nouns

stanza_cache = {}
def replace_variation(caption, word, variation):
    """Replace the original noun phrase with variation"""
    sent = stanza_cache.get(caption)
    if not sent:
        sent =  STANZA_MODEL(caption)
        stanza_cache[caption] = sent

    tree = sent.sentences[0].constituency
    parts = extract_nouns(tree)
    parts = [variation + ' ' if word in c else c for c in parts]
    parts = ''.join(parts)
    parts = re.sub(r',\s+,', ',', parts)
    return parts

def generate_variation_captions(values):
    """Generates new captions"""
    new_captions = []
    item = values['item']
    t2t_caption = values['tag2text_caption']
    short = values['short']

    if item in t2t_caption:
        new_captions.append(replace_variation(t2t_caption, item, short))

    for caption in values['original_captions']:
        if item in caption:
            new_captions.append(replace_variation(item, caption, short))

    if len(new_captions) < 1:
        new_captions.append(f'{t2t_caption} and {short}')

    new_train_point = {**values, 'captions' : new_captions}
    return new_train_point

def main(args):
    """Main function find samples and process them."""
    df = pd.read_json(args.annotation_file)
    df = df[df['save'].apply(lambda x: Path(x).exists())].reset_index(drop=True)
    # save_num = len(df) // 1000
    logging.info('Total %d samples', len(df))

    output_file = args.output_dir / args.output_file
    dataset = []
    pbar = tqdm(df.to_dict(orient='records'))
    for values in pbar:
        new_point = generate_variation_captions(values)
        dataset.append(new_point)
    pd.DataFrame(dataset).to_json(output_file, orient='records')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", default='variations.json')
    parser.add_argument("--output-dir", default='../output', type=Path)
    parser.add_argument("--output-file", default='variation_dataset.json', type=Path)

    _args = parser.parse_args()

    main(_args)
