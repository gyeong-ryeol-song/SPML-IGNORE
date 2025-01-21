import os
import traceback
from config import get_configs
from train_q2l import run_train_q2l
from train_resnet import run_train_resnet
import torch

def main():
    
    args = get_configs()
    print(args, '\n')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'train_q2l':
        print('## Train Q2L Start ##') 
        run_train_q2l(args)
    elif args.mode == 'train_resnet':
        print('## Train ResNet Start ##') 
        run_train_resnet(args)  
     

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())

