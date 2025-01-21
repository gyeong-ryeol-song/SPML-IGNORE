import torch
import torch.nn.functional as F

def cam_minmax_norm(cam):
    
    cam_min = cam.min(dim=0)[0]
    cam_max = cam.max(dim=0)[0]
    cam_img = (cam - cam_min) / cam_max
    
    return cam_img

def generate_cam_and_mask(image, atts, cnn_feat, label_vec, args, device):
    
    # Make CAM
    atts = atts.view(cnn_feat.size(0), args.num_classes, cnn_feat.size(2), cnn_feat.size(3))
    atts_sp = atts[label_vec.type(torch.bool), :, :] 
    atts_sp = atts_sp.unsqueeze(1)  
    
    idx = label_vec.sum(dim=1) == 1 
    cnn_feat_subset = cnn_feat[idx] if idx.any() else cnn_feat  
    cnn_feat = (cnn_feat_subset * atts_sp).mean(dim=1)  

    # CAM Normalization
    cam = cam_minmax_norm(cnn_feat)  # Min-Max 
    
    # CAM Interpolation
    try:
        cam = F.interpolate(cam.unsqueeze(1), size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False) 
    except:
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False) 

    cam_mask = torch.where(cam > args.cam_threshold, 1, 0).float()  
    cam_mask_mod = torch.ones(image.size(0), 1, image.size(-2), image.size(-1)).to(device) 
    
    cam_mask_mod[idx] = cam_mask  
    # Masked Image
    masked_image = image * cam_mask_mod  
    
    return masked_image

def generate_resnet_cam_and_mask(image, cam_w, label_vec, args, device):

    if label_vec.sum() == 0:
        cam_mask_mod = torch.ones(image.size(0), 1, image.size(-2), image.size(-1)).to(device)
        masked_image = image * cam_mask_mod
        return masked_image, cam_mask_mod
    
    # CAM mormalization
    cam_sp = cam_w[label_vec.type(torch.bool), :, :]
    cam = cam_minmax_norm(cam_sp)

    # interpolation
    try:
        cam = F.interpolate(cam.unsqueeze(1), size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
    except:
        cam = cam.unsqueeze(1)
        cam = F.interpolate(cam, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)

    idx = label_vec.sum(dim=1) == 1  

    cam_mask = torch.where(cam > args.cam_threshold, 1, 0).float()  
    cam_mask_mod = torch.ones(image.size(0), 1, image.size(-2), image.size(-1)).to(device)  
    cam_mask_mod[idx] = cam_mask 
    # Masked Image
    masked_image = image * cam_mask_mod

    return masked_image
