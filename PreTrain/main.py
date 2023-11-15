from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
import torch.nn.functional as F
from src import train

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def main():
    epoch = 40
    batch_size = 161
    learning_rate = 5e-3
    state_config = {'embed_dim': 12,
                   # vision
                   'image_resolution' : 2048,
                   'vision_layers' : [3, 4, 6, 3],
                   'vision_width' : 64,
                   'vision_patch_size' : 7,
                   # text
                   'context_length' : 14,
                   'vocab_size' : 32, # 大一点啊
                   'transformer_width' : 128,  # 512 大一点啊
                   'transformer_heads' : 8,
                   'transformer_layers' : 3
                   }

    #     image_path ='/kaggle/input/coffee-bean-dataset-resized-224-x-224/train/'
    image_path = './data/12kNoSample/total'
    train.train_fine_tuning(state_config, epoch, batch_size, learning_rate, image_path, param_group=False)


if __name__ == '__main__':
    main()