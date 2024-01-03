<h1 align="center">Generative Negative Mining</h1>
<p align="center">
  <a href="https://ugorsahin.github.io/enhancing-multimodal-compositional-reasoning-of-vlm.html"> Project Page </a> |
  <a href="https://huggingface.co/ugursahin/generative-negative-mining-clip"> Model Checkpoint </a> |
  <a href="https://huggingface.co/datasets/ugursahin/generative-negative-mining-dataset"> Dataset </a> 
</p>


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

## Resources
[Training annotation file (.json)](https://huggingface.co/datasets/ugursahin/generative-negative-mining-dataset/resolve/main/train.json?download=true)

[Training images (.zip)](https://cvg.cit.tum.de/webshare/g/papers/khamuham/comp_reason/file.zip)

[Pretrained model (pytorch)](https://huggingface.co/ugursahin/generative-negative-mining-clip)

## Citation
If you find our work helpful in your research, please consider citing us
```latex

@misc{sahin2023enhancing,
      title={Enhancing Multimodal Compositional Reasoning of Visual Language Models with Generative Negative Mining}, 
      author={Ugur Sahin and Hang Li and Qadeer Khan and Daniel Cremers and Volker Tresp},
      year={2023},
      eprint={2311.03964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      journal = {Winter Conference on Applications of Computer Vision},
}
```
## Acknowledgments

- [Tag2Text](https://github.com/xinyu1205/recognize-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Stanza](https://github.com/stanfordnlp/stanza)
- [BLIP](https://github.com/salesforce/BLIP)
