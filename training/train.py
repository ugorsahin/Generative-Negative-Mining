from pathlib import Path

import clip
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from helpers import count_parameters
from negative_mining_dataset import prepare_dataset

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--epoch',                  default=10)
    parser.add_argument('--mode',                   default='item_based')
    parser.add_argument('--device',                 default='cuda:0')
    parser.add_argument('--model-checkpoint',       default=None)
    parser.add_argument('--save-path', type=Path,   default='../test_folder/model_save_folder')
    parser.add_argument('--dataset',                default='../test_folder/variation_dataset_164k.json')
    parser.add_argument('--image-root', type=Path,  default='../test_folder/variations')
    parser.add_argument('--coco-dataset',           default='../test_folder/dataset/coco_karpathy_train_combined.json')
    parser.add_argument('--coco-image_dir',         default='../test_folder/dataset')
    parser.add_argument('--sample-by-score',        action='store_true')
    parser.add_argument('--clip-model',             default="ViT-B/32")
    parser.add_argument('--lr',                     default=1e-6)
    parser.add_argument('--beta-1',                 default=0.9)
    parser.add_argument('--beta-2',                 default=0.98)
    parser.add_argument('--adam-eps',               default=1e-6)
    parser.add_argument('--weight-decay',           default=0.1)

    args = parser.parse_args()

    if not args.device:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    EPOCH = args.epoch
    args.save_path.mkdir(exist_ok=True)

    model, preprocess = clip.load(
        args.clip_model,
        device=args.device,
        jit=False
    )
    model.train()
    print(f'There are #{count_parameters(model)} trainable parameters.')
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay
    )

    dataloader = prepare_dataset(args, preprocess)

    loss_img = CrossEntropyLoss()
    loss_txt = CrossEntropyLoss()

    LAST_EPOCH = 0
    if args.model_checkpoint:
        checkpoint = torch.load(args.model_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        LAST_EPOCH = checkpoint['epoch'] + 1
        print(f'loaded checkpoint from {args.model_checkpoint}')

    print(f'There are #{EPOCH} epochs.')
    for epoch in range(LAST_EPOCH, LAST_EPOCH+EPOCH):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        all_losses      = []

        for batch in pbar:
            optimizer.zero_grad()

            images, texts, _ = batch

            images = images.to(args.device)
            texts = texts.to(args.device)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=args.device)
            logits_im, logits_text = model(images, texts)
            loss = (loss_img(logits_im, ground_truth) + loss_txt(logits_text, ground_truth))/2

            loss.backward()
            optimizer.step()

            pbar.set_postfix({'cross loss': loss.item()})
            all_losses.append(loss.item())

        print(f'Cross: {np.mean(all_losses):.4}')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            args.save_path / f'model_{epoch}.pt'
        )
