import torch
import torch.nn as nn
import torch.optim as optim
import logging
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.utils import set_seed, resample_data, spike_to_counts2, save_to_excel, loadAllDays, AllDays_split
from src.utils import load_mat, spike_to_counts1, save_data2txt, gaussian_nomalization
from src.model import GRU
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, Subset, DataLoader
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "GRU"
dataFile = "indy_20160627_01.mat"
dataPath = "../data/Makin/"
npy_folder_path = "../data/Makin_processed_npy"
ori_npy_folder_path = "../data/Makin_origin_npy/"
excel_path = 'results/'
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10  # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 32
nEpoch = 50
gap_num = 10  # the time slice
seq_size = 1024  # the length of the sequence
input_size = 96
hidden_size = 256
out_size = 2  # the output dim
num_layers = [1,2] # np.arange(2,11)

# learning rate
lrInit = 6e-4 if modelType == "GRU" else 4e3  # Transormer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "GRU" else 0.01
epochLengthFixed = 10000  # make every epoch very short, so we can see the training progress
dimensions = ['num_layers', 'test_r2', 'test_loss', 'train_r2', 'train_loss']

# loading data
print('loading data... ' + ori_npy_folder_path)


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

results = []

prefix = 'Scaling_multisetion'

s_train, s_test, t_train, t_test = AllDays_split(ori_npy_folder_path)

train_Dataset = Dataset(seq_size, out_size, s_train, t_train)
test_Dataset = Dataset(seq_size, out_size, s_test, t_test)

src_feature_dim = train_Dataset.x.shape[1]
trg_feature_dim = train_Dataset.y.shape[1]

train_dataloader = DataLoader(train_Dataset, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_Dataset, batch_size=len(test_Dataset), shuffle=True)
for num_layer in num_layers:
    # setting the model parameters
    gru_layer = nn.GRU(input_size, hidden_size, num_layer, batch_first=True)
    model = GRU(input_size, hidden_size, out_size,
                # num_layer, gru_layer,
                gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, gru_layer.bias_ih_l0,
                gru_layer.bias_hh_l0)
    # model = gru_layer(input_size, hidden_size, num_layer, out_size)


    # from thop import profile
    # input1 = torch.randn((1, 4, 128, 96))
    # flops, params = profile(model, inputs=input1.to('cuda'))
    # print('Macs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-3)

    print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
          'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layer)

    tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                          learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                          warmupTokens=0, finalTokens=nEpoch * len(train_Dataset) * seq_size, numWorkers=0,
                          epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                          out_size=out_size, seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers,
                          criterion=criterion, optimizer=optimizer)

    trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
    trainer.train()
    result = trainer.test()
    result['name'] = prefix
    result['num_layers'] = num_layer
    results.append(result)
    print(prefix + 'done')
        # torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        #            + '.pth')
    save_to_excel(results, excel_path + prefix + '-' + str(
        nEpoch) + '-' + modelType + '-' + 'results.xlsx', modelType, nEpoch, dimensions)

