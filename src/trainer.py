import math
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from torcheval.metrics.functional import r2_score

from src.utils import save_data2txt

logger = logging.getLogger(__name__)


class TrainerConfig:
    maxEpochs = 10
    batchSize = 32
    learningRate = 4e-3
    betas = (0.9, 0.99)
    eps = 1e-8
    gradNormClip = 1.0
    weightDecay = 0.01
    lrDecay = False
    warmupTokens = 375e6
    finalTokens = 260e9
    epochSaveFrequency = 0
    epochSavePath = 'trained-'
    numWorkers = 0
    warmup_steps = 4000
    total_steps = 10000


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.t = False
        self.config = config
        self.avg_test_loss = 0
        self.tokens = 0     # counter used for learning rate decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("当前设备:", self.device)
        current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("当前GPU设备的名称:", current_gpu_name)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

    def get_runName(self):
        rawModel = self.model.module if hasattr(self.model, "module") else self.model
        cfg = self.config
        runName = (cfg.modelType) + str(cfg.seq_size) + '-' + str(cfg.hidden_size) + '-' + str(cfg.out_dim)

        return runName

    def train_epoch(self, split, epoch, model, config):
        predicts = []
        targets = []
        totalLoss = 0
        totalR2s = 0
        self.t = split == 'train'
        model.train(self.t)
        data = self.train_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True, batch_size=config.batchSize,
                            num_workers=config.numWorkers)

        pbar = tqdm(enumerate(loader), total=len(loader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if self.t else enumerate(loader)

        for it, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(self.t):
                out = model(x)
                predicts.append(out.view(-1, 2))
                targets.append(y.view(-1, 2))
                # loss = loss.mean()

                if self.t:
                    model.zero_grad()
                    loss = self.config.criterion(out.view(-1, 2), y.view(-1, 2))
                    r2_s = r2_score(out.view(-1, 2), y.view(-1, 2))
                    totalLoss += loss.item()
                    totalR2s += r2_s
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradNormClip)
                    self.config.optimizer.step()

                    if config.lrDecay:
                        self.tokens += (y >= 0).sum()
                        lrFinalFactor = config.lrFinal / config.learningRate
                        if self.tokens < config.warmupTokens:
                            # linear warmup
                            lrMult = lrFinalFactor + (1 - lrFinalFactor) * float(self.tokens) / float(
                                config.warmupTokens)
                            progress = 0
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmupTokens) / float(
                                max(1, config.finalTokens - config.warmupTokens))
                            # progress = min(progress * 1.1, 1.0) # more fine-tuning with low LR
                            lrMult = (0.5 + lrFinalFactor / 2) + (0.5 - lrFinalFactor / 2) * math.cos(
                                math.pi * progress)

                        lr = config.learningRate * lrMult
                        for paramGroup in self.config.optimizer.param_groups:
                            paramGroup['lr'] = lr
                    else:
                        lr = config.learningRate

                    pbar.set_description(
                        f"epoch {epoch+1} progress {progress * 100.0:.2f}% iter {it + 1}: r2_score "
                        f"{totalR2s / (it + 1):.2f} loss {totalLoss / (it + 1):.4f} lr {lr:e}")
        
        with open("train.csv", "a", encoding="utf-8") as file:
            file.write(f"{totalLoss / (it + 1):.4f}, {totalR2s / (it + 1):.4f}\n")

    def train(self):
        model, config = self.model, self.config
        with open("train.csv", "a", encoding="utf-8") as file:
            file.write(f"train average loss, train average r2 score\n")

        for epoch in range(config.maxEpochs):
            predicts, targets = self.train_epoch('train', epoch, model, config)
            # print(self.avg_train_loss / len(self.train_dataset))

            if (config.epochSaveFrequency > 0 and epoch % config.epochSaveFrequency == 0) or (epoch ==
                                                                                              config.maxEpochs - 1):
                # DataParallel wrappers keep raw model object in .module
                rawModel = self.model.module if hasattr(self.model, "module") else self.model
                torch.save(rawModel, self.config.epochSavePath + str(epoch + 1) + '.pth')

            # save the model predicts and targets every 10 epoch
            # if (epoch + 1) % config.epochSaveFrequency == 0:
            #     save_data_to_txt(predicts, targets, 'predict_character.txt', 'target_character.txt')

    def test(self):
        model, config = self.model, self.config
        model.eval()

        predicts = []
        targets = []
        self.t = False
        model.train(self.t)
        data = self.test_dataset
        totalLoss = 0
        totalR2s = 0
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=config.batchSize,
                            num_workers=config.numWorkers)

        pbar = tqdm(enumerate(loader), total=len(loader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if self.t else enumerate(loader)
        ct = 0
        for it, (x, y) in pbar:
            x = x.to(self.device)  # place data on the correct device
            y = y.to(self.device)
            ct += 1

            with torch.set_grad_enabled(self.t):
                out = model(x)  # forward the model
                predicts.append(out.view(-1, 2))
                targets.append(y.view(-1, 2))
                # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                loss = self.config.criterion(out.view(-1, 2), y.view(-1, 2))
                r2_s = r2_score(out.view(-1, 2), y.view(-1, 2))
            totalLoss += loss.item()
            totalR2s += r2_s
            print(f"Batch Loss: {loss:.4f} R2_score: {r2_s:.4f}")
        # predicts = []
        # targets = []
        with open("train.csv", "a", encoding="utf-8") as file:
            file.write(f"test loss, test r2 score\n")

        pbar = enumerate(self.test_dataloader)
        with torch.no_grad():
            for ct, (data, target) in pbar:
                data = data.to(self.device)
                target = target.to(self.device)
                out = model(data)  # forward the model

                # predicts.append(out.detach().cpu().view(-1, 2))
                # targets.append(target.detach().cpu().view(-1, 2))

                loss = self.config.criterion(out.view(-1, 2), target.view(-1, 2))
                r2_s = r2_score(out.view(-1, 2), target.view(-1, 2))
                
                print(f"Test Mean Loss: {loss:.4f}, R2_score: {r2_s:.4f},  Num_iter: {ct}")
                
                with open("train.csv", "a", encoding="utf-8") as file:
                    file.write(f"{loss:.4f}, {r2_s:.4f}\n")

        # save_data2txt(predicts, 'src_trg_data/test_predict.txt')
        # save_data2txt(targets, 'src_trg_data/test_target.txt')

        print(f"Test Mean Loss: {totalLoss / ct:.4f}, R2_score: {totalR2s / ct:.4f},  Num_iter: {ct}")

        # save_data2txt(predicts, 'src_trg_data/test_predict.txt')
        # save_data2txt(targets, 'src_trg_data/test_target.txt')
