import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from instrumentation import compute_metrics
import losses
import os
from functions import *
from tqdm import tqdm
import query2labels._init_paths
from query2labels.lib.models_q2.query2label import build_q2l
from torch.optim import lr_scheduler

def run_train_q2l(args):
    dataset = datasets.get_data(args)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=args.bsize,
            shuffle=(phase == 'train'),
            sampler=None,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True
        )
    
    args.num_class = args.num_classes
    model = build_q2l(args)
    model = model.cuda()

    args.lr_mult = args.bsize / 256
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError
    
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        steps_per_epoch=len(dataloader["train"]), 
        epochs=args.num_epochs, 
        pct_start=0.2
    )

    bestmap_val = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Thresholds
    Avg_threshold = torch.zeros(1, args.num_classes).to(device)
    Gap_threshold = 0

    for epoch in range(1, args.num_epochs + 1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                train_loss = 0
                batch_stack = 0
            else:
                model.eval()
                y_pred = np.zeros((len(dataset[phase]), args.num_classes))
                y_true = np.zeros((len(dataset[phase]), args.num_classes))
                batch_stack = 0

            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    image = batch['image'].to(device, non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                    label_vec_true = batch['label_vec_true'].clone().numpy()

                    optimizer.zero_grad()
                    logits, cnn_feat_w, atts_w, _ = model(image)

                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)

                    if phase == 'train':
                        image_strong = batch['image_strong'].to(device, non_blocking=True)
                        logits_s, cnn_feat_s, atts_s, _ = model(image_strong)
                        img_w = image.clone().detach()
                        
                        masked_image = generate_cam_and_mask(img_w, atts_w, cnn_feat_w, label_vec_obs, args, device)
                        mask_logits, cnn_feat_m, atts_m, _ = model(masked_image)
                        
                        gap = torch.abs(logits - mask_logits).clone().detach()
                        logits_cp = logits.clone().detach()
                        Gap_threshold = args.gap_lambda * Gap_threshold + (1 - args.gap_lambda) * gap[label_vec_obs.type(torch.bool)].mean()
                        Avg_threshold = args.avg_lambda * Avg_threshold + (1 - args.avg_lambda) * (logits_cp.sum(dim=0) / batch["label_vec_obs"].size(0))

                        loss_tensor = losses.ignore_loss(
                            logits, mask_logits, logits_s, label_vec_obs, args, Gap_threshold, Avg_threshold, epoch
                        )
                        loss_tensor.backward()
                        optimizer.step()
                        scheduler.step()
                        train_loss += loss_tensor.clone().detach().cpu().numpy()
                    else:
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack: batch_stack + this_batch_size] = preds_np
                        y_true[batch_stack: batch_stack + this_batch_size] = label_vec_true
                        batch_stack += this_batch_size

            if phase != 'train':
                metrics = compute_metrics(y_pred, y_true)
                map_val = metrics['map']

        print(f"Epoch {epoch}: val mAP {map_val:.3f}\n")
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            print(f"Saving model weight for best val mAP {bestmap_val:.3f}")
            path = os.path.join(args.save_path, "{}_{}_bestmodel.pt".format(args.dataset, args.mode))
            torch.save((model.state_dict(), args), path)

    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)
    model = model.cuda()
    phase = 'test'
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), args.num_classes))
    y_true = np.zeros((len(dataset[phase]), args.num_classes))
    batch_stack = 0

    with torch.set_grad_enabled(False):
        for batch in tqdm(dataloader[phase]):
            image = batch['image'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            logits, _, _, _ = model(image)

            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            
            preds = torch.sigmoid(logits)
            preds_np = preds.cpu().numpy()
            
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack: batch_stack + this_batch_size] = preds_np
            y_true[batch_stack: batch_stack + this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']
    print(f"Training procedure completed!")
    print(f"Test mAP: {map_test:.3f} when trained until epoch {bestmap_epoch}")

    