import os
import json

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CombinedCocoDataset(Dataset):
    def __init__(
            self,
            transform,
            image_root,
            file_path,
            max_words=30,
            prompt=''
        ):

        with open(file_path, 'r', encoding='utf-8') as fd:
            self.annotation = json.load(fd)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids:
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root,ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = ann['captions']
        caption = caption[np.random.choice(np.arange(len(caption)))]

        return image, caption, self.img_ids[ann['image_id']]
