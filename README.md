# Generative-Negative-Mining

This is the official implementation for the paper 
[ Enhancing Multimodal Compositional Reasoning of Visual Language Models with Generative Negative Mining](). 
We propose a framework that not only mines in both directions but also generates challenging negative samples in both modalities, i.e., images and texts. Leveraging these generative hard negative samples, we significantly enhance VLMs’ performance in tasks involving multimodal compositional reasoning.

![Overview of the framework](https://ugorsahin.github.io/static/genemi.png)
## Installation
Requirements: python>=3.8, cuda=11.3
```
git clone https://github.com/ugorsahin/Generative-Negative-Mining
cd Generative-Negative-Mining
pip install -r requirements.txt
```
We advise installing on a virtual environment to avoid library dependency crashes.

## Data Generation
The semi-synthetic variations can be generated stage-by-stage or by using the pipeline script (if you don't have enough gpu memory)

To run the pipeline
- Change into directory `generation_pipeline` by using `cd generation_pipeline`
- Run the script as follows
```shell
python pipeline.py \
--tag2text-checkpoint=<path/to/tag2text_model> \
--gd-config=<path/to/gd_config> \
--gd-checkpoint=<path/to/gd_checkpoint> \
--sam-checkpoint=<path/to/sam_checkpoint> \
--sd-checkpoint=<path/to/sd_checkpoint> \
--output-dir=<path/to/output> \
--input-dir=<path/to/images> \
--root-dir=<path/to/root>
```

## Train
To train the clip
- Change into directory `training` by using `cd training`
- Run the following code.
```shell
python train.py 
--epoch=<number_of_epochs> \
--mode='allinone|item_based|image_based' \
--save-path=<path/to/save_folder> \
--dataset=<path/to/variation_dataset> \
--image-root=<path/to/image_root> \
--coco-dataset=<path/to/coco_dataset> \
--coco-image_dir=<path/to/coco_images> \
```

## Evaluation
To evaluate the model checkpoints
- - Change into directory `evaluation` by using `cd evaluation`
```shell
python eval_autogmentation.py \
--model-name=<tag-for-your-model> \
--snapshot_file=<specify if you want to evaluate one model>' \
--snapshot_folder=<specify if you want to evaluate all training>'
--evaluation-filepath=<path/to/evaluation_annotations> \
--evaluation-image-folder=<path/to/eval/images>
```
- Either set snapshot-file or snapshot-folder

## Citation
If you find our work helpful in your research, please consider citing us
```latex
@article{sahin2024enhancing,
    author    = {Ugur Sahin, Hang Li, Qadeer Khan, Daniel Cremers, Volker Tresp},
    title     = {Enhancing Multimodal Compositional Reasoning of Visual Language Models with Generative Negative Mining},
    journal   = {WACV},
    year      = {2024},
}
```
## Acknowlegments

- [Tag2Text](https://github.com/xinyu1205/recognize-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Stanza](https://github.com/stanfordnlp/stanza)
- [BLIP](https://github.com/salesforce/BLIP)
