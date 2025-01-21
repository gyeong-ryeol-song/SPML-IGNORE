import numpy as np
import torch
import torch.nn.functional as F
import datasets
import models
from instrumentation import compute_metrics
import losses
import os
from functions import *
from tqdm import tqdm

def run_train_resnet(args):
    dataset = datasets.get_data(args)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=args.bsize,
            shuffle=phase == 'train',
            sampler=None,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True
        )

    model = models.ResNet_1x1(args)
    
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    onebyone_conv_params = [param for param in list(model.onebyone_conv.parameters()) if param.requires_grad]
    opt_params = [
        {'params': feature_extractor_params, 'lr': args.lr},
        {'params': onebyone_conv_params, 'lr': args.lr_mult * args.lr}
    ]
  
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    bestmap_val = 0

    # logit gap threshold & Logit Average Threshold
    Avg_threshold = torch.zeros(1, args.num_classes).to(device, non_blocking=True)
    Gap_threshold = 0

    for epoch in range(1, args.num_epochs + 1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                train_loss = 0
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
                    label_vec_true_ = batch['label_vec_true'].to(device, non_blocking=True)

                    # Forward pass
                    optimizer.zero_grad()
                    logits, cam_w = model(image)
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)

                    if phase == 'train':
                        image_strong = batch['image_strong'].to(device, non_blocking=True)
                        logits_s, _ = model(image_strong)
                        
                        masked_image = generate_resnet_cam_and_mask(image, cam_w, label_vec_obs, args, device)
                        # import pdb;pdb.set_trace()
                        mask_logits, _ = model(masked_image)
                        gap = torch.abs(logits - mask_logits).clone().detach()
                        logits_cp = logits.clone().detach()

                        # compute logit gap threshold & ensured candidate selection threshold
                        Gap_threshold = args.gap_lambda * Gap_threshold + (1 - args.gap_lambda) * gap[label_vec_obs.type(torch.bool)].mean()
                        Avg_threshold = args.avg_lambda * Avg_threshold + (1 - args.avg_lambda) * (logits_cp.sum(dim=0) / batch["label_vec_obs"].size(0))

                        # compute batch loss
                        loss_tensor = losses.ignore_loss(
                            logits, mask_logits, logits_s, label_vec_obs, args, Gap_threshold, Avg_threshold, epoch
                        )
                        loss_tensor.backward()
                        optimizer.step()
                    else:
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack:batch_stack + this_batch_size] = preds_np
                        y_true[batch_stack:batch_stack + this_batch_size] = label_vec_true
                        batch_stack += this_batch_size

            if phase != 'train':
                metrics = compute_metrics(y_pred, y_true)
                del y_pred
                del y_true
                map_val = metrics['map']

        print(" Epoch {} : val mAP {:.3f} \n".format(epoch, map_val))
        
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
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

    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(image)
            
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)

            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack:batch_stack + this_batch_size] = preds_np
            y_true[batch_stack:batch_stack + this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']

    print('Training procedure completed!')
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')