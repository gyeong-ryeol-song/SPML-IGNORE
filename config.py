import os
import sys
import argparse
from os.path import join as ospj


_DATASET = ('pascal', 'coco', 'nuswide', 'cub')
_TRAIN_SET_VARIANT = ('observed', 'clean')
_OPTIMIZER = ('adam', 'sgd')
_SCHEMES = ('LL-R', 'LL-Ct', 'LL-Cp', 'None')
_LOOKUP = {
    'feat_dim': {
        'resnet50': 2048
    },
    'num_classes': {
        'pascal': 20,
        'coco': 80,
        'nuswide': 81,
        'cub': 312
    }
}



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_dir(runs_dir, exp_name):
    runs_dir = ospj(runs_dir, exp_name)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    return runs_dir

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss_seed', type=int, default=999,
                        help='seed fo subsampling')
    parser.add_argument('--ss_frac_train', type=float, default=1.0,
                        help='fraction of training set to subsample')
    parser.add_argument('--ss_frac_val', type=float, default=1.0,
                        help='fraction of val set to subsample')
    parser.add_argument('--use_feats', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='False if end-to-end training, True if linear training')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--split_seed', type=int, default=1200)
    parser.add_argument('--train_set_variant', type=str, default='observed',
                        choices=_TRAIN_SET_VARIANT)
    parser.add_argument('--val_set_variant', type=str, default='clean')
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--freeze_feature_extractor', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--use_pretrained', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--exp_name', type=str, default='exp_default')
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # parser.add_argument("--ls_coef", type=float, default=0.1)
    # parser.add_argument("--gap_threshold", type=float, default=3.0)
    # parser.add_argument('--portion', type=float, default = 0.8)
    # parser.add_argument('--world_size',type = int, default = 4)
    # parser.add_argument('--DDP_backend',type=str,default="nccl")
        # distribution training
    # parser.add_argument('--world-size', default=-1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='env://', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--seed', default=None, type=int,
    #                     help='seed for initializing training. ')
    # parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    
    #dataset
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=_DATASET)
    parser.add_argument('--img_size',type=int, default=448)
    
    # Model Training Hyperparameters
    parser.add_argument('--mode', type=str, default = 'train_q2l', choices = ['train_q2l',"train_resnet"])
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=_OPTIMIZER)
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')
    parser.add_argument('--bsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_mult', type=float, default=10)
    parser.add_argument('--mod_scheme', type=str, default='None', 
                        choices=_SCHEMES)
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('--warm_up', type = int, default = 3)
    parser.add_argument('--cam_threshold', type = float, default = 0.8)
    parser.add_argument("--gap_lambda", type = float, default = 0.999)
    parser.add_argument("--avg_lambda", type = float, default = 0.999)
    
    # loss weight
    parser.add_argument("--reg_weight",type = float, default = 0.5)
    parser.add_argument("--spbc_weight",type = float, default = 0.5)
    parser.add_argument("--ignore_weight",type = float, default = 1.)
    
    # losses
    parser.add_argument('--strong_aug',type=str2bool,default=True)
    parser.add_argument('--ignore',type=str2bool,default=True)
    parser.add_argument('--spbc',type=str2bool,default=True)
    parser.add_argument('--lat',type=str2bool,default=True)
    parser.add_argument('--reg',type=str2bool,default=True)
    
    #transformer
    parser.add_argument('--num_class', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
    
    args = parser.parse_args()
    args.feat_dim = _LOOKUP['feat_dim'][args.arch]
    args.num_classes = _LOOKUP['num_classes'][args.dataset]
    args.save_path = set_dir(args.save_path, args.exp_name)
    
    return args


