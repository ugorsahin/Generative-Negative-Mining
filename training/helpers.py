"""Utility functions for training phase"""

import re

def pre_caption(caption,max_words=50):
    'Prepares the caption for tokenizer'
    caption = caption.lower()
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption)
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    caption = ' '.join(caption.split(' ')[:max_words])
    return caption

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)