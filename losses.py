import torch
import torch.nn.functional as F

LOG_EPSILON = 1e-5

'''
helper functions
'''

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def log_loss(preds, targs):
    return targs * neg_log(preds)

'''
loss functions
'''

def loss_an(preds, label_vec_obs):
    observed_labels = label_vec_obs
    assert torch.min(observed_labels) >= 0

    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    
    return loss_mtx


def loss_spbc(preds, label_vec_obs):
    observed_labels = label_vec_obs
    assert torch.min(observed_labels) >= 0
    loss_mtx = neg_log(preds[observed_labels == 1])
    
    return loss_mtx

def ignore_loss(logits, mask_logits, logits_st, label_vec, args, Gap_Threshold, Avg_Threshold, epoch): 
    
    preds = torch.sigmoid(logits)
    preds_s = torch.sigmoid(logits_st)

    gap = torch.abs(logits - mask_logits).detach()
    
    # rejection function
    if args.lat:
        pseudo = ((gap > Gap_Threshold) & (logits > Avg_Threshold)).type(torch.float32).detach()
    else:
        pseudo = ((gap > Gap_Threshold)).type(torch.float32).detach()
    
    # warm up phase
    if epoch <= args.warm_up:
        mask = torch.zeros_like(label_vec)
    else:
        mask = (pseudo - label_vec).clip(max=1.0, min=0.0)
        
    # false negative rejection 
    fn_mask = (torch.ones_like(mask) - mask).detach()
    
    # losses
    loss_spbc_mtx = loss_spbc(preds, label_vec)
    loss_spbc_ = loss_spbc_mtx.mean()
    consistency_reg = F.mse_loss(logits_st, logits)
    
    # strong augmentation
    if args.strong_aug:
        loss_mtx = loss_an(preds_s, label_vec)
    else:
        loss_mtx = loss_an(preds, label_vec)
    
    # Ignore Loss
    if args.ignore:
        loss_ignore = loss_mtx[fn_mask.type(torch.bool)].mean()
    else:
        loss_ignore = loss_mtx.mean()
    
    # SPBC
    if not args.spbc:
        args.spbc_weight = 0.0
    
    # reg and overall_loss
    if args.reg:
        overall_loss = args.spbc_weight * loss_spbc_ + args.ignore_weight * loss_ignore + args.reg_weight * consistency_reg 
    else:
        overall_loss = args.spbc_weight * loss_spbc_ + args.ignore_weight * loss_ignore
    
    return overall_loss