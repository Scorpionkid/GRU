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
csv_file = "result/train_multiSectionTest.csv"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth-trained-"
batchSize = 32
nEpoch = 50
seq_size = 1024    # the length of the sequence
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


spike_trainList, spike_testList, target_trainList, target_testList, section_name = AllDays_split(dataPath)
spike_train = np.concatenate(spike_trainList, axis=0)
target_train = np.concatenate(target_trainList, axis=0)
train_dataset = Dataset(seq_size, out_size, spike_train, target_train)
train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=batchSize, num_workers=numWorkers)

spike_test = np.concatenate(spike_testList, axis=0)
target_test = np.concatenate(target_testList, axis=0)
test_dataset = Dataset(seq_size, out_size, spike_test, target_test)
test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=len(test_dataset),
                         num_workers=numWorkers)

src_feature_dim = train_dataset.x.shape[1]
trg_feature_dim = train_dataset.y.shape[1]

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize,
      'seq_size', seq_size, 'hidden_size', hidden_size, 'num_layers', num_layers)

with open(csv_file, "a", encoding="utf-8") as file:
    file.write(dataPath + "batch size" + str(batchSize) + "epochs" + str(nEpoch) + "\n")

# setting the model parameters
gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)
model = GRU(input_size, hidden_size, out_size, gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, gru_layer.bias_ih_l0, gru_layer.bias_hh_l0)
rawModel = model.module if hasattr(model, "module") else model
rawModel = rawModel.float()

print("number of parameters: " + str(sum(p.numel() for p in model.parameters())) + "\n")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(rawModel.parameters(), lr=4e-3)


print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'seq_size', seq_size, 'hidden_size', hidden_size,
      'num_layers', num_layers)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps, warmupTokens=0,
                      finalTokens=nEpoch*len(train_dataset)*seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath, out_size=out_size,
                      seq_size=seq_size, hidden_size=hidden_size, num_layers=num_layers, criterion=criterion,
                      optimizer=optimizer, csv_file=csv_file, out_dim=out_size)

trainer = Trainer(model, None, None, tConf)

with open(csv_file, "a", encoding="utf-8") as file:
    file.write(f"section name, test loss, test r2 score, run time\n")
trainer.train(train_loader)

trainer.test(test_loader, "Test ALL")

for i, (test_spike, test_target) in enumerate(zip(spike_testList, target_testList)):
    testDataset = Dataset(seq_size, out_size, test_spike, test_target)
    testLoader = DataLoader(testDataset, shuffle=False, pin_memory=True, batch_size=batchSize, num_workers=numWorkers)
    trainer.test(testLoader, section_name[i])

# torch.save(model, epochSavePath + trainer.get_runName() + '-' +
#            datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')

