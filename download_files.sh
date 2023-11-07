#!/bin/bash

if ! command -v wget &>/dev/null; then
    echo "wget is not found. Please install it."
    exit 1

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P model_lib
wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth -P model_lib
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P model_lib
