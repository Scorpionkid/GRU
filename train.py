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
from torch.utils.data import Dataset, DataLoader
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
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 128
nEpoch = 50
seq_size = 256    # the length of the sequence
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


# class Dataset(Dataset):
#     def __init__(self, ctx_len, vocab_size, spike, target):
#         print("loading data...", end=' ')
#         self.ctxLen = ctx_len
#         self.vocabSize = vocab_size
#         self.x = spike
#         self.y = target
#
#         # Gaussian normalization
#         # self.x, self.y = gaussian_nomalization(x, y)
#
#         # min-max normalization
#         # self.x, self.y = min_max_nomalization(x, y)
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, item):
#         # i = np.random.randint(0, len(self.x) - self.ctxLen)
#         i = item % (len(self.x) - self.ctxLen)
#         x = torch.tensor(self.x[i:i + self.ctxLen, :], dtype=torch.float32)
#         y = torch.tensor(self.y[i:i + self.ctxLen, :], dtype=torch.float32)
#         # 用于测试的简化版本
#         # x = torch.randn(self.ctxLen, 96)  # 假设数据形状为[ctxLen, 96]
#         # y = torch.randn(self.ctxLen, 2)  # 假设标签形状为[ctxLen, 2]
#         return x, y


class Dataset(Dataset):
    def __init__(self, ctx_len, vocab_size, spike, target):
        print("loading data...", end=' ')
        self.ctxLen = ctx_len
        self.vocabSize = vocab_size
        self.x, self.y = Reshape_ctxLen(spike, target, ctx_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


s_train, s_test, t_train, t_test = AllDays_split(dataPath)
train_dataset = Dataset(seq_size, out_size, s_train, t_train)
test_dataset = Dataset(seq_size, out_size, s_test, t_test)


src_feature_dim = train_dataset.x.shape[1]
trg_feature_dim = train_dataset.y.shape[1]


# 按时间连续性划分数据集
# trainSize = int(0.8 * len(dataset))
# train_Dataset, test_Dataset = split_dataset(ctxLen, out_dim, dataset, trainSize)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batchSize, pin_memory=True)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset), pin_memory=True)

# setting the model parameters
gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)
model = GRU(input_size, hidden_size, out_size, gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, gru_layer.bias_ih_l0,
            gru_layer.bias_hh_l0)
rawModel = model.module if hasattr(model, "module") else model
rawModel = rawModel.float()

print("number of parameters: " + str(sum(p.numel() for p in model.parameters())))

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(rawModel.parameters(), lr=4e-3)


print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
      'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layers)


tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_dataset)*seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_size, seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers,
                      criterion=criterion, optimizer=optimizer)

trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
trainer.train()
trainer.test()

# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#            + '.pth')

with open("train.csv", "a", encoding="utf-8") as file:
    file.write(dataPath + "batch size" + str(batchSize) + "epochs" + str(nEpoch) + "\n")

spike_train, spike_test, target_train, target_test = AllDays_split(dataPath)
train_dataset = Dataset(spike_train, target_train, seq_size, out_size)
test_dataset = Dataset(spike_test, target_test, seq_size, out_size)

train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=batchSize)
test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=len(test_dataset))


src_feature_dim = train_dataset.x.shape[-1]
trg_feature_dim = train_dataset.y.shape[-1]


# setting the model parameters
gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)
model = GRU(input_size, hidden_size, out_size, gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, gru_layer.bias_ih_l0, gru_layer.bias_hh_l0)
rawModel = model.module if hasattr(model, "module") else model
rawModel = rawModel.float()

print("number of parameters: " + str(sum(p.numel() for p in model.parameters())) + "\n")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(rawModel.parameters(), lr=4e-3)


print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
        'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layers)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                        learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                        warmupTokens=0, finalTokens=nEpoch*len(train_dataset)*seq_size, numWorkers=0,
                        epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                        out_size=out_size, seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers,
                        criterion=criterion, optimizer=optimizer)

trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
trainer.train()
result = trainer.test()

# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#            + '.pth')

