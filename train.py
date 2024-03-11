import torch
import torch.nn as nn
import torch.optim as optim
import logging
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.utils import set_seed, resample_data, spike_to_counts2
from src.utils import load_mat, spike_to_counts1, save_data2txt, gaussian_nomalization
from src.model import GRU
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, random_split, Subset
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "GRU"
dataFile = "indy_20160624_03.mat"
dataPath = "data/"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 8
nEpoch = 50
gap_num = 10    # the time slice
seq_size = 64    # the length of the sequence
input_size = 96
hidden_size = 256
out_size = 2   # the output dim
num_layers = 2

# learning rate
lrInit = 6e-4 if modelType == "GRU" else 4e3   # Transormer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "GRU" else 0.01
epochLengthFixed = 10000    # make every epoch very short, so we can see the training progress


# loading data
print('loading data... ' + dataFile)


class Dataset(Dataset):
    def __init__(self, seq_size, out_size, spike, target):
        print("loading data...", end=' ')
        self.seq_size = seq_size
        self.out_size = out_size
        self.x = spike
        self.y = target

    def __len__(self):
        # 向上取整
        return (len(self.x) + self.seq_size - 1) // self.seq_size

    def __getitem__(self, idx):
        # i = np.random.randint(0, len(self.x) - self.ctxLen)
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

# class Dataset_list(Dataset):
#     def __init__(self, ctx_len, vocab_size, x, y):
#         self.ctxLen = ctx_len
#         self.vocabSize = vocab_size
#         self.x = x
#         self.y = y
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         x = self.x[idx]
#         y = self.y[idx]
#         return x, y

def split_dataset(ctxLen, out_size, dataset, train_size):
    test_size = len(dataset) - train_size
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, len(dataset)))

    train_x = dataset.x[train_indices]
    train_y = dataset.y[train_indices]
    save_data2txt(train_x, 'src_trg_data/train_spike_num.txt')
    save_data2txt(train_y, 'src_trg_data/train_target_velocity.txt')

    test_x = dataset.x[test_indices]
    test_y = dataset.y[test_indices]
    save_data2txt(test_x, 'src_trg_data/test_spike_num.txt')
    save_data2txt(test_y, 'src_trg_data/test_target_velocity.txt')

    train_dataset = Dataset(ctxLen, out_size, train_x, train_y)
    test_dataset = Dataset(ctxLen, out_size, test_x, test_y)

    return train_dataset, test_dataset

spike, y, t = load_mat(dataPath+dataFile)
# y = resample_data(y, 4, 1)
# new_time = np.linspace(t[0, 0], t[0, -1], len(y))
# spike, target = spike_to_counts2(spike, y, np.transpose(new_time), gap_num)
spike, target = spike_to_counts1(spike, y, t[0])
# spike = np.transpose(spike)

# spike = np.load('data/indy_20160622_01_processed_spike.npy')
# target = np.load('data/indy_20160622_01_processed_target.npy')

dataset = Dataset(seq_size, out_size, spike, target)

# 归一化
dataset.x, dataset.y = gaussian_nomalization(dataset.x, dataset.y)
# 平滑处理
dataset.x = gaussian_filter1d(dataset.x, 3, axis=0)
dataset.y = gaussian_filter1d(dataset.y, 3, axis=0)

src_feature_dim = dataset.x.shape[1]
trg_feature_dim = dataset.y.shape[1]


# 按时间连续性划分数据集
# trainSize = int(0.8 * len(dataset))
# train_Dataset, test_Dataset = split_dataset(ctxLen, out_size, dataset, trainSize)
train_Dataset = Subset(dataset, range(0, int(0.8 * len(dataset))))
test_Dataset = Subset(dataset, range(int(0.8 * len(dataset)), len(dataset)))

# setting the model parameters
gru_layer = nn.GRU(input_size, hidden_size, batch_first = True)
model = GRU(input_size, hidden_size, out_size, gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, gru_layer.bias_ih_l0, gru_layer.bias_hh_l0)
rawModel = model.module if hasattr(model, "module") else model
rawModel = rawModel.float()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(rawModel.parameters(), lr=4e-3)


print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
      'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layers)


tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_Dataset)*seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_size=out_size, seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers,
                      criterion=criterion, optimizer=optimizer)

trainer = Trainer(model, train_Dataset, test_Dataset, tConf)
trainer.train()
trainer.test()

torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
           + '.pth')
