import torch
import torch.nn as nn
import torch.optim as optim
import logging
import datetime
import numpy as np
from src.utils import set_seed, resample_data, spike_to_counts2
from src.utils import load_mat, spike_to_counts1, save_data2txt, gaussian_nomalization
from src.utils import *
from src.model import GRU
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, DataLoader, Subset
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "GRU"
dataFile = "Makin"
dataPath = "../Makin/Makin_origin_npy/"
csv_file = "train_Section27finetune.csv"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 8
nEpoch = 10
seq_size = 128    # the length of the sequence
input_size = 96
hidden_size = 256
out_size = 2   # the output dim
num_layers = 2

# learning rate
lrInit = 6e-4 if modelType == "GRU" else 4e3   # Transormer can use higher learning rate
lrFinal = 4e-4
numWorkers = 0

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "GRU" else 0.01
epochLengthFixed = 10000    # make every epoch very short, so we can see the training progress

# loading data
print('loading data... ' + dataFile)


class Dataset(Dataset):
    def __init__(self, ctx_len, vocab_size, spike, target):
        self.ctxLen = ctx_len
        self.vocabSize = vocab_size
        self.x, self.y = Reshape_ctxLen(spike, target, ctx_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


spike, target = loadAllDays(dataPath)
dataset = Dataset(seq_size, out_size, spike, target)
trainLen = int(0.8 * len(dataset))
testDataset = Subset(dataset, range(trainLen, len(dataset)))

model = torch.load('pth-trained-GRU1024-256-2-2024-04-14-10-43-35.pth')
print("number of parameters: " + str(sum(p.numel() for p in model.parameters())) + "\n")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=4e-3)

test_loader = DataLoader(testDataset, shuffle=False, pin_memory=True, num_workers=numWorkers,
                         batch_size=len(testDataset))

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps, warmupTokens=0,
                      finalTokens=nEpoch*len(dataset)*seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath, out_size=out_size,
                      eq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers, criterion=criterion,
                      optimizer=optimizer, csv_file=csv_file, out_dim=out_size)

trainer = Trainer(model, None, None, tConf)

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'seq_size', seq_size, 'hidden_size', hidden_size,
      'num_layers', num_layers)

with open(csv_file, "a", encoding="utf-8") as file:
    file.write(dataPath + "batch size" + str(batchSize) + "epochs" + str(nEpoch) + "\n")

trainer.test(test_loader, "Test ALL")

for step in range(0, trainLen - batchSize, batchSize):
    trainDataset = Subset(dataset, range(step, step + batchSize))

    if trainDataset:
        train_loader = DataLoader(trainDataset, shuffle=True, pin_memory=True, num_workers=numWorkers,
                                  batch_size=batchSize)
    else:
        train_loader = None

    trainer.train(train_loader)

    trainer.test(test_loader, "Test ALL")
