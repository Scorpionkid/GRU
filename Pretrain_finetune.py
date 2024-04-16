import os
import torch.nn as nn
import random
import logging
import datetime
import numpy as np
from torch import optim

from Transformer.src.utils import loadAllDays
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
batchSize = 32
nEpoch = 30
modelLevel = "word"  # "character" or "word"
seq_size = 1024  # the length of the sequence
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


# dividing the dataset into training set and testing dataset by a ratio of 82
spike, target, section_name = loadAllDays(ori_npy_folder_path)
spike = np.concatenate(spike, axis=0)
target = np.concatenate(target, axis=0)

prefix = 'finetune_day1~25'

dataset = Dataset(seq_size, out_size, spike, target)

src_feature_dim = dataset.x.shape[1]
trg_feature_dim = dataset.y.shape[1]
# setting the model parameters
gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)
model = GRU(input_size, hidden_size, out_size, gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, gru_layer.bias_ih_l0,
            gru_layer.bias_hh_l0)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=4e-3)

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
      'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layers)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch * len(dataset) * seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_size=out_size, seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers,
                      criterion=criterion, optimizer=optimizer)

train_dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, pin_memory=True)
trainer = Trainer(model, train_dataloader, None, tConf)
trainer.train()
print(prefix + 'done')

torch.save(model, epochSavePath + prefix + '-' + trainer.get_runName() + '-' +
           datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
           + '.pth')

