import argparse
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import build_sam, SamPredictor
from tqdm import tqdm

import grounding_dino as gd
import create_tags as t2t
import portray_variations as chat
import generate_variation_images as ra
import score_dataset as score
import rewrite_caption_stanza as stn

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.WARNING
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size",             default=384)
    parser.add_argument("--tag2text-threshold",     default=0.64)
    parser.add_argument("--save-annotated-image",   default=False, action='store_true')
    parser.add_argument("--tag2text-checkpoint",    default='../model_lib/tag2text_swin_14m.pth')
    parser.add_argument("--gd-config",              default="../model_lib/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gd-checkpoint",          default="../model_lib/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam-checkpoint",         default='../model_lib/sam_vit_h_4b8939.pth')
    parser.add_argument("--sd-checkpoint",          default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument("--sd-num-steps",           default=100)
    parser.add_argument("--sd-guidance",            default=7.5)
    parser.add_argument("--blip-model",             default="blip_image_text_matching")
    parser.add_argument("--device",                 default='cuda')
    parser.add_argument("--output-dir", type=Path,  default='outputs')
    parser.add_argument("--input-dir",  type=Path,  default='assets')
    parser.add_argument("--annotation-file",        default=None)
    parser.add_argument("--root-dir",   type=Path,  default='../', )
    parser.add_argument("--verbose",                action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.annotation_file:
        images = t2t.prepare_annotation_file(args.root_dir, args.annotation_file)
    else:
        images = t2t.prepare_image_folder(args.root_dir, args.input_dir)

    assert len(images) > 0, 'At least one image is required to run the pipeline'

    ann_path = None
    args.input_dir = args.root_dir / args.input_dir
    args.output_dir = args.root_dir / args.output_dir
    imgout_dir = args.output_dir / 'variations'

    if args.save_annotated_image:
        ann_path = args.input_dir / 'annotations'
        ann_path.mkdir(exists_ok=True)

    # LOAD ALL MODELS
    tag2text_model, transform = t2t.load_model(
        pretrained=args.tag2text_checkpoint,
        image_size=args.image_size,
        threshold=args.tag2text_threshold,
        vit = 'swin_b',
        device=_device
    )
    logging.info('Tag2Text model is ready')
    gd_model = gd.load_model()
    logging.info('Grounding DINO is ready')
    # SET YOUR ENV VARIABLES
    # ENV VARIABLES = HUGGINGCHAT_UNAME, HUGGINGCHAT_PWD
    chathead = chat.HuggingChatClient(
        verbose=args.verbose
    )
    # chathead.switch_model('meta-llama/Llama-2-70b-chat-hf')
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_checkpoint))
    logging.info('Sam is ready')
    sd_model =  StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_checkpoint,
        torch_dtype=torch.float16,
    )
    sd_model.set_progress_bar_config(leave=False)
    sd_model = sd_model.to("cuda")
    logging.info('Stable diffusion is ready')
    blip_model, blip_vis, blip_txt = score.load_blip_model(args.blip_model, _device)
    itm_func = lambda x: score.process_image(
        model=blip_model,
        vis_pre=blip_vis,
        text_pre=blip_txt,
        device=_device,
        captions=x['captions'],
        image_path=imgout_dir / x['save'],
    )

    logging.info('BLIP is ready')
    # END OF LOAD ALL MODELS

    all_dataset = pd.DataFrame()
    FIRST_RUN = True
    for item in tqdm(images):

        image_path = str(item['_image_source'])
        save_ann_path = ann_path / Path(item['image']).name if ann_path else None

        tags, tag2text_caption, _ = t2t.extract_tags(
            tag2text_model      = tag2text_model,
            gd_model            = gd_model,
            tag2text_transform  = transform,
            image_path          = image_path,
            device              =_device,
            save_annotation     = save_ann_path
        )

        if len(tags) <= 0:
            logging.info('Tag2Text found 0 tags in the image %s, continuing', image_path)
            continue

        image_info = {
            'image'             : item['image'],
            'original_captions' : item['original_captions'],
            '_image_source'     : item['_image_source'],
            'tag2text_caption'  : tag2text_caption,
            'items'             : tags
        }

        if FIRST_RUN:
            answer = chathead.interact(chat.input_query)
            FIRST_RUN = False
            logging.info('Chathead is ready')

        response, parsed_response = chat.process_one_item(chathead, image_info)
        if len(parsed_response) <= 0:
            logging.info(
                'The output of ChatGPT is not parsed in the image %s, continuing', image_path)
            continue

        image_info.update({
            'llm_output' : response,
            'variations': parsed_response
        })

        inpaint_df = ra.create_generation_df(image_list=[image_info])
        inpaint_df = inpaint_df.apply(stn.generate_variation_captions, axis=1, result_type='expand')

        groups = [(k, _df) for k, _df in inpaint_df.groupby(['origin_name', 'source', 'item'])]
        random.shuffle(groups)
        pbar = tqdm(groups)
        
        iva_scores = []
        for (origin_name, source, item), samples in pbar:
            _attr = {'file': origin_name, 'item' : item}
            pbar.set_postfix(_attr)
            output_im_dir = imgout_dir / origin_name / item
            output_im_dir.mkdir(parents=True, exist_ok=True)
            image_input = args.input_dir / source
            ra.generate_variations(
                gd_model, sam_model, sd_model, image_input, samples,
                image_folder=imgout_dir,
                num_inference_steps=args.sd_num_steps,
                guidance_scale=args.sd_guidance
            )
            iva_score = score.image_variation_area(imgout_dir / samples.save)
            iva_scores.extend([iva_score] * 4)

        inpaint_df = inpaint_df.apply(stn.generate_variation_captions, axis=1, result_type='expand')
        inpaint_df['itm_score'] = inpaint_df.apply(itm_func, axis=1, result_type='expand')
        inpaint_df['iva_score'] = iva_scores

        all_dataset = pd.concat([all_dataset, inpaint_df], axis=0)
        all_dataset.to_json(args.output_dir / 'variation_annotation.json', orient='records')
