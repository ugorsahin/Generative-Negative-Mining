"""Training code for clip"""

import json

from clip import tokenize
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from coco_karpathy_dataset import CombinedCocoDataset
from helpers import pre_caption

def collate(batch):
    image = torch.stack([k for i in batch for k in i[0]])
    text = tokenize([k for i in batch for k in i[1]])
    _id = [i[2] for i in batch]
    return image, text, _id

def prepare_dataset(args, preprocess):
    if args.mode == 'allinone':
        group_keys          = ['name']
        coco_image_per_item = 1
        var_per_item        = 1
        sample_by_score     = False
        batch_size          = 100 # 100 * (var_per_item (1) + coco_image_per_item (1) ) = 400 samples per iter
    elif args.mode == 'item_based':
        group_keys          = ['origin_name', 'item']
        coco_image_per_item = 8
        var_per_item        = 2
        sample_by_score     = True
        batch_size          = 20 # 40 * (var_per_item (2) + coco_image_per_item (8) ) = 400 samples per iter
    elif args.mode == 'image_based':
        group_keys = ['origin_name', 'item']
        coco_image_per_item = 8
        var_per_item        = 2
        sample_by_score     = True
        batch_size          = 20 # 40 * (var_per_item (2) + coco_image_per_item (8) ) = 400 samples per iter

    train_dataset = CombinedDataset(
        preprocess,
        image_root          = args.image_root,
        sample_by_score     = sample_by_score,
        group_keys          = group_keys,
        variation_ann       = args.dataset,
        coco_image_per_item = coco_image_per_item,
        var_per_item        = var_per_item,
        coco_dataset        = args.coco_dataset,
        coco_image_folder   = args.coco_image_dir
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        num_workers = 1,
        pin_memory  = True,
        sampler     = None,
        shuffle     = True,
        collate_fn  = collate,
        drop_last   = True
    )
    return train_dataloader

class CombinedDataset(Dataset):
    def __init__(
            self,
            transform,
            image_root,
            max_words           = 30,
            prompt              = '',
            variation_ann       = None,
            itm_score_thres     = 0,
            iva_score_thres     = 14,
            prob_eps            = 0.5,
            var_per_item        = 2,
            coco_image_per_item = 6,
            group_keys          = None,
            sample_by_score     = False,
            seed_num            = None,
            coco_dataset        = '../dataset_folder/coco_dataset',
            coco_image_folder   = '../dataset_folder/coco_dataset'
        ):

        with open(variation_ann, 'r', encoding='utf-8') as fd:
            self.annotation = json.load(fd)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.img_ids = {}
        self.num_variation = 0
        self.augments = {}
        self.sample_by_score = sample_by_score
        self.probs = {}
        self.var_per_item = var_per_item
        self.coco_image_per_item = coco_image_per_item

        self.coco_dataset = CombinedCocoDataset(transform, coco_dataset, coco_image_folder)
        self.coco_size = len(self.coco_dataset)
        self.ctr = 0
        self.randoms = None
        self.rng = np.random.default_rng(seed_num)
        self.build_random_gen()

        if group_keys is None:
            group_keys = ['name']

        self.annotation = list(filter(
            lambda x: np.mean(x['itm_score']) > itm_score_thres and x['iva_score'] > iva_score_thres,
            self.annotation
        ))

        origin_ims, origin_ims_items = set(), set()

        for variation in self.annotation:
            self.num_variation += 1
            origin_ims.add(variation['origin_name'])
            origin_ims_items.add(f'{variation["origin_name"]}:{variation["item"]}')

            item_id = '_'.join([variation[key] for key in group_keys])

            if item_id not in self.augments:
                self.augments[item_id] = [variation]
            else:
                self.augments[item_id].append(variation)

            itm_score_mean = np.mean(variation['itm_score'])
            if item_id not in self.probs:
                self.probs[item_id] = [itm_score_mean]
            else:
                self.probs[item_id].append(itm_score_mean)

        for key in list(self.probs.keys()):
            prob_array = self.probs[key]
            prob_array = np.array(prob_array)
            prob_array = prob_array - prob_array.min() + prob_eps
            prob_array /= prob_array.sum()
            self.probs[key] = prob_array

        print(f'There are #{len(origin_ims)} augmented origins')
        print(f'There are #{len(origin_ims_items)} items')
        print(f'There are #{self.num_variation} variations')
        self.augments = [(k, _v) for k, _v in self.augments.items()]

    def build_random_gen(self):
        self.ctr = 0
        self.randoms = self.rng.choice(np.arange(self.coco_size), self.coco_size, replace=False)

    def get_coco_item(self):
        if self.ctr >= len(self.randoms):
            self.build_random_gen()
            print(self.randoms[:10])

        item = self.coco_dataset[self.randoms[self.ctr]]
        self.ctr+=1
        return item

    def __len__(self):
        return len(self.augments)

    def __getitem__(self, index):
        _id, ann = self.augments[index]
        arange = np.arange(len(ann))
        if len(ann) <= self.var_per_item:
            pass
        elif self.sample_by_score:
            try:
                selected = np.random.choice(arange, self.var_per_item, p=self.probs[_id], replace=False)
                ann = list(map(lambda x: ann[x], selected))
            except Exception as err:
                print(err)
                np.random.shuffle(arange)
                ann = [ann[i] for i in arange[:self.var_per_item]]
        else:
            np.random.shuffle(arange)
            ann = [ann[i] for i in arange[:self.var_per_item]]

        _id = [i['name'] for i in ann]
        image_paths = [self.image_root / item['source'] for item in ann]
        images = [
            self.transform(Image.open(image_path).convert('RGB'))
            for image_path in image_paths
        ]

        captions = list(
            map(
                lambda x: pre_caption(x['captions'][np.random.choice(len(x['captions']))]),
                ann
            )
        )
        random_samples = list(map(lambda _: self.get_coco_item(), range(self.coco_image_per_item)))
        images = images + list(map(lambda x: x[0], random_samples))
        captions = captions + list(map(lambda x: x[1], random_samples))
        _id = _id + list(map(lambda x: x[2], random_samples))
        return images, captions, _id
