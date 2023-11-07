'''This script is responsible for the utility function for generation process.'''
import json
import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np

def prepare_image_folder(root_folder, image_folder):
    '''
        Creates a dictionary of items from given image image folder.
    '''
    image_data = (root_folder / image_folder).glob('*.jpg')
    image_data = [
        {
            'image' : str(im_path.name),
            '_image_source' : im_path,
            'original_captions' : []
        } for im_path in image_data
    ]
    logging.info('Total #%d images', len(image_data))
    return image_data

def prepare_annotation_file(root_folder, annotation_file):
    '''
    This function takes an annotation file and creates a list of items to process.
    The annotation file should be a json file that uses records orientation and each record should have the keys
    'caption', 'image' and 'image_id'
    '''
    image_dict = {}
    with open(annotation_file, 'r', encoding='utf-8') as fd:
        annotation_dict = json.load(fd)

    for item in annotation_dict:
        key = item['image']
        if image_dict[key]:
            image_dict['original_captions'].append(item['caption'])
        else:
            image_dict[key] = {
                'image' : item['image_id'],
                '_image_source' : str(root_folder / item['image']),
                'original_captions' : item['caption']
            }

    image_data = list(image_dict.values())
    return image_data

def create_generation_df(df=None, image_list=None, save_path=None):
    '''
        Creates rows of generation sample covering the all information about samples.
    '''
    _df = []
    if image_list is None:
        image_list = df.to_dict(orient='records')
    for image in image_list:
        im_name = re.search(r'(.+)\.jpg', image['image']).group(1)
        for item, portrayals in image['variations'].items():
            for portre in portrayals:
                regex_match = re.search(r'(.+)\((.*)\)', portre)
                if not regex_match:
                    continue
                _, short_version = regex_match.groups()
                sv_path = re.sub(r'[/ ]', '_', short_version)

                _df.append({
                    'origin_name' : im_name,
                    'name' : f'{im_name}:{item}:{sv_path}',
                    'source' : image['image'],
                    'item'  : item,
                    'short' : short_version,
                    'save'  : str(Path(im_name) / item / f'{sv_path}.jpg'),
                    'tag2text_caption' : image['tag2text_caption'],
                    'original_captions' : image['original_captions']
                })
    out = pd.DataFrame(_df)
    logging.info('Total #%d variations', len(out))
    if save_path:
        out.to_json(save_path, orient='records')
    return out

def find_square(arr):
    '''
        Finds the area that has the item in it
        If any dimension is smaller than 512, the dimension is extended.
    '''
    H, W = arr.shape
    y, x = np.nonzero(arr)
    ys, ye = y.min(), y.max()
    xs, xe = x.min(), x.max()
    if ye - ys < 512:
        ycenter = y.min() + (y.max() - y.min()) // 2
        y_start, y_end = ycenter - 256, ycenter + 256
        if y_start < 0:
            y_end -= y_start
            y_start = 0
        if y_end > H:
            y_start -= (y_end - H)
            y_end = H
        y_start = np.maximum(0, y_start)
        y_end = np.minimum(arr.shape[0], y_end)
    else:
        y_start = ys
        y_end = ye

    if xe - xs < 512:
        xcenter = x.min() + (x.max() - x.min()) // 2
        x_start, x_end = xcenter - 256, xcenter + 256
        if x_start < 0:
            x_end -= x_start
            x_start = 0
        if x_end > W:
            x_start -= (x_end - W)
            x_end = W
        x_start = np.maximum(0, x_start)
        x_end = np.minimum(arr.shape[1], x_end)
    else:
        x_start = xs
        x_end = xe

    return x_start, x_end, y_start, y_end
