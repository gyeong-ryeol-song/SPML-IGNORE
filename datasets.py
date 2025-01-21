import os
import json
import numpy as np
from PIL import Image
import torch
import copy
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms

def get_metadata(dataset_name):
    if dataset_name == 'pascal':
        meta = {
            'num_classes': 20,
            'path_to_dataset': 'data/pascal',
            'path_to_images': 'data/pascal/VOCdevkit/VOC2012/JPEGImages',
            'path_to_seg_labels': 'data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        }
    elif dataset_name == 'coco':
        meta = {
            'num_classes': 80,
            'path_to_dataset': 'data/coco',
            'path_to_images': 'data/coco'
        }
    elif dataset_name == 'nuswide':
        meta = {
            'num_classes': 81,
            'path_to_dataset': 'data/nuswide',
            'path_to_images': 'data/nuswide/Flickr'
        }
    elif dataset_name == 'cub':
        meta = {
            'num_classes': 312,
            'path_to_dataset': 'data/cub',
            'path_to_images': 'data/cub/CUB_200_2011/images'
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta

def get_imagenet_stats():
    '''
    Returns standard ImageNet statistics. 
    '''
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    return (imagenet_mean, imagenet_std)

def get_transforms(args):
    '''
    Returns image transforms.
    '''
    (imagenet_mean, imagenet_std) = get_imagenet_stats()
    tx = {}
    tx['train'] = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['val'] = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    tx['test'] = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    tx['raw'] = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    return tx

def generate_split(num_ex, frac, rng):
    '''
    Computes indices for a randomized split of num_ex objects into two parts,
    so we return two index vectors: idx_1 and idx_2. Note that idx_1 has length
    (1.0 - frac)*num_ex and idx_2 has length frac*num_ex. Sorted index sets are 
    returned because this function is for splitting, not shuffling. 
    '''
    
    # compute size of each split:
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2
    
    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])

    return (idx_1, idx_2)

def get_data(args):
    
    '''
    Given a parameter dictionary args, initialize and return the specified dataset. 
    '''
    
    # define transforms:
    tx = get_transforms(args)
    
    # select and return the right dataset:
    if args.dataset == 'coco':
        ds = multilabel(args, tx).get_datasets()
    elif args.dataset == 'pascal':
        ds = multilabel(args, tx).get_datasets()
    elif args.dataset == 'nuswide':
        ds = multilabel(args, tx).get_datasets()
    elif args.dataset == 'cub':
        ds = multilabel(args, tx).get_datasets()
    else:
        raise ValueError('Unknown dataset.')
    
    # Optionally overwrite the observed training labels with clean labels:
    # assert args.train_set_variant in ['clean', 'observed']
    if args.train_set_variant == 'clean':
        print('Using clean labels for training.')
        ds['train'].label_matrix_obs = copy.deepcopy(ds['train'].label_matrix)
    else:
        print('Using single positive labels for training.')
    
    # Optionally overwrite the observed val labels with clean labels:
    assert args.val_set_variant in ['clean', 'observed']
    if args.val_set_variant == 'clean':
        print('Using clean labels for validation.')
        ds['val'].label_matrix_obs = copy.deepcopy(ds['val'].label_matrix)
    else:
        print('Using single positive labels for validation.')
    
    # We always use a clean test set:
    ds['test'].label_matrix_obs = copy.deepcopy(ds['test'].label_matrix)
            
    return ds

def load_data(base_path, args):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, f'formatted_{phase}_labels.npy'))
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, f'formatted_{phase}_labels_obs.npy'))
        data[phase]['images'] = np.load(os.path.join(base_path, f'formatted_{phase}_images.npy'))
        data[phase]['feats'] = np.load(getattr(args, f'{phase}_feats_file')) if args.use_feats else []
    return data

class GetAffinityLabelFromIndices():
    
    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

   
class SPML_train_dataset(Dataset):
    
    def __init__(self, dataset_name, image_ids, label_matrix, label_matrix_obs, feats, tx, tx_raw, use_feats):
        meta = get_metadata(dataset_name)
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.feats = feats
        self.tx = tx
        self.tx_raw = tx_raw
        self.tx_st = copy.deepcopy(tx)
        self.tx_st.transforms.insert(0, transforms.RandAugment(3, 5))
        self.use_feats = use_feats

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        if self.use_feats:
            # Set I to be a feature vector:
            I = torch.FloatTensor(np.copy(self.feats[idx, :]))
            
        else:
            # Set I to be an image: 
            image_path = os.path.join(self.path_to_images, self.image_ids[idx])

            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
                I_og = self.tx_raw(I_raw.convert('RGB'))
                I_strong = self.tx_st(I_raw.convert('RGB'))
        out = {
            'image': I,
            "image_raw" : I_og,
            'image_strong' : I_strong,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
            # 'image_path': image_path # added for CAM visualization purpose
        }
        
        return out 

class SPML_test_dataset(Dataset):
    
    def __init__(self, dataset_name, image_ids, label_matrix, label_matrix_obs, feats, tx, tx_raw, use_feats):
        meta = get_metadata(dataset_name)
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        
        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.feats = feats
        self.tx = tx
        self.tx_raw = tx_raw
        self.use_feats = use_feats

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        if self.use_feats:
            # Set I to be a feature vector:
            I = torch.FloatTensor(np.copy(self.feats[idx, :]))
            I_np = np.copy(self.feats[idx, :])
        else:
            # Set I to be an image: 
            image_path = os.path.join(self.path_to_images, self.image_ids[idx])
            with Image.open(image_path) as I_raw:
                I = self.tx(I_raw.convert('RGB'))
                I_np = self.tx_raw(I_raw.convert('RGB'))
        out = {
            'image': I,
            'image_raw' : I_np,
            'label_vec_obs': torch.FloatTensor(np.copy(self.label_matrix_obs[idx, :])),
            'label_vec_true': torch.FloatTensor(np.copy(self.label_matrix[idx, :])),
            'idx': idx,
            # 'image_path': image_path # added for CAM visualization purpose
        }
        
        return out
    
def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    return (category_list, id_to_index)

class multilabel:
    
    def __init__(self, args, tx):

        # get dataset metadata:
        meta = get_metadata(args.dataset)
        self.base_path = meta['path_to_dataset']
        
        # load data:
        source_data = load_data(self.base_path, args)
        
        # generate indices to split official train set into train and val:
        split_idx = {}
        (split_idx['train'], split_idx['val']) = generate_split(
            len(source_data['train']['images']),
            args.val_frac,
            np.random.RandomState(args.split_seed)
        )
        
        # subsample split indices:
        ss_rng = np.random.RandomState(args.ss_seed)
        temp_train_idx = copy.deepcopy(split_idx['train'])
        for phase in ['train', 'val']:
            num_initial = len(split_idx[phase])
            num_final = int(np.round(getattr(args, f'ss_frac_{phase}') * num_initial))  # Fix here
            split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]
        
        # define train set:
        if args.mode == "train_q2l":
            self.train = SPML_train_dataset( 
                args.dataset,
                source_data['train']['images'][split_idx['train']],
                source_data['train']['labels'][split_idx['train'], :],
                source_data['train']['labels_obs'][split_idx['train'], :],
                source_data['train']['feats'][split_idx['train'], :] if args.use_feats else [],
                tx['train'],
                tx['raw'],
                args.use_feats
            )
        elif args.mode == "train_resnet":
            self.train = SPML_train_dataset( 
                args.dataset,
                source_data['train']['images'][split_idx['train']],
                source_data['train']['labels'][split_idx['train'], :],
                source_data['train']['labels_obs'][split_idx['train'], :],
                source_data['train']['feats'][split_idx['train'], :] if args.use_feats else [],
                tx['train'],
                tx['raw'],
                args.use_feats
            ) 
        else:
            self.train = SPML_test_dataset(
                args.dataset,
                source_data['val']['images'],
                source_data['val']['labels'],
                source_data['val']['labels_obs'],
                source_data['val']['feats'],
                tx['test'],
                tx['raw'],
                args.use_feats
            ) 
        
        # define val set:
        self.val = SPML_test_dataset(
            args.dataset,
            source_data['train']['images'][split_idx['val']],
            source_data['train']['labels'][split_idx['val'], :],
            source_data['train']['labels_obs'][split_idx['val'], :],
            source_data['train']['feats'][split_idx['val'], :] if args.use_feats else [],
            tx['val'],
            tx["raw"],
            args.use_feats
        )
        
        # define test set:
        self.test = SPML_test_dataset(
            args.dataset,
            source_data['val']['images'],
            source_data['val']['labels'],
            source_data['val']['labels_obs'],
            source_data['val']['feats'],
            tx['test'],
            tx['raw'],
            args.use_feats
        )
        
        # define dict of dataset lengths: 
        self.lengths = {'train': len(self.train), 'val': len(self.val), 'test': len(self.test)}
    
    def get_datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}
