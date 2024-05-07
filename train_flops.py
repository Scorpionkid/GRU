import os
import torch.nn as nn
import random
import logging
import datetime
import numpy as np
from torch import optim

from src.utils import *
from src.model import GRU
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import Dataset, random_split, Subset
import os

set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "GRU"
dataFile = "indy_20160627_01.mat"
dataPath = "../data/Makin/"
npy_folder_path = "../data/Makin_processed_npy"
ori_npy_folder_path = "../data/Makin_origin_npy"
csv_file = "results/shuffleDaySequenceTrainAll.csv"
excel_path = 'results/'
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transformer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10  # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 8
nEpoch = 10
nEpochs = [3, 5, 7, 10]
modelLevel = "word"  # "character" or "word"
seq_size = 128  # the length of the sequence
input_size = 96
hidden_size = 256
out_size = 2  # the output dim
num_layers = 2

# learning rate
lrInit = 6e-4 if modelType == "GRU" else 4e3  # Transormer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "GRU" else 0.01
epochLengthFixed = 10000  # make every epoch very short, so we can see the training progress
dimensions = ['timestep/s', 'test_r2', 'test_loss', 'train_r2', 'train_loss']


class Dataset(Dataset):
    def __init__(self, seq_size, out_size, spike, target):
        print("loading data...", end=' ')
        self.seq_size = seq_size
        self.out_size = out_size
        self.x = spike
        self.y = target

    def __len__(self):
        return (len(self.x) + self.seq_size - 1) // self.seq_size

    def __getitem__(self, idx):
        start_idx = idx * self.seq_size
        end_idx = start_idx + self.seq_size

        # 处理最后一个可能不完整的序列
        if end_idx > len(self.x):
            # 对x和y进行填充以达到期望的序列长度
            pad_size_x = end_idx - len(self.x)
            x_padded = np.pad(self.x[start_idx:len(self.x), :], ((0, pad_size_x), (0, 0)), mode='constant',
                              constant_values=0)
            y_padded = np.pad(self.y[start_idx:len(self.y), :], ((0, pad_size_x), (0, 0)), mode='constant',
                              constant_values=0)
            x = torch.tensor(x_padded, dtype=torch.float32)
            y = torch.tensor(y_padded, dtype=torch.float32)
        else:
            x = torch.tensor(self.x[start_idx:end_idx, :], dtype=torch.float32)
            y = torch.tensor(self.y[start_idx:end_idx, :], dtype=torch.float32)
        return x, y


import torch, time
model = torch.load('pth/trained-finetune_day1~25-GRU1024-256-2-2024-04-16-02-13-31.pth')
from thop import profile
input1 = torch.randn(128, 128, 96).to('cuda')
start_time = time.time()
flops, params = profile(model, inputs=(input1,))
end_time = time.time()
print(f"Time: {end_time - start_time:.4f}")
print('Macs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')


