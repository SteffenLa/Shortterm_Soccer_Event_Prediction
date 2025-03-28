from multiprocessing import cpu_count
from models.abstract import ClassificationModel
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils.imbalanced_sampler import ImbalancedDatasetSampler
import pytorch_lightning as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from evaluation.metrics import find_best_threshold


class ConvNet(pl.LightningModule):
    def __init__(self, in_channels, class_weights, lr, dropout_probability=0.4, use_batchnorm=True, classification=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(*[
            # CONV 1
            nn.Conv1d(in_channels, 16, (5,)),
            nn.BatchNorm1d(16) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout_probability),
            nn.MaxPool1d(2, 2),
            # CONV 2
            nn.Conv1d(16, 16, (5,)),
            nn.BatchNorm1d(16) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout_probability),
            nn.MaxPool1d(2, 2),
            # CONV 3
            nn.Conv1d(16, 32, (5,)),
            nn.BatchNorm1d(32) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout_probability),
            # CONV 4
            nn.Conv1d(32, 16, (5,)),
            nn.BatchNorm1d(16) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout_probability),
            # CONV 5
            nn.Conv1d(16, 32, (5,)),
            nn.BatchNorm1d(32) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout_probability),
            nn.MaxPool1d(2, 2, ceil_mode=True),
            # LINEAR 1
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_probability),
            # LINEAR 2
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Dropout(dropout_probability),
            # LINEAR 3
            nn.Linear(16, 1),
            nn.Sigmoid() if classification else nn.Identity()
        ])
        self.lr = lr
        self.hyperparameters = kwargs
        self.class_weight = class_weights
        self.classification = classification

    def forward(self, x):
        assert x.shape[-1] == 120, "Convolutional Model is built for fixed size input of 120 frames"
        return self.model(x)

    def training_step(self, batch, *args, **kwargs):
        data = batch[0]
        labels = batch[1].float()

        pred = self.forward(data)
        pred = torch.squeeze(pred, 1)
        if self.classification:
            loss = nn.BCELoss(
                weight=torch.tensor([self.class_weight[0] if l == 0 else self.class_weight[1] for l in labels]))
        else:
            loss = nn.MSELoss()
        self.log("train_loss", loss(pred, labels))
        return loss(pred, labels)

    def validation_step(self, batch, *args, **kwargs):
        data = batch[0]
        labels = batch[1].float()
        pred = self.forward(data)
        pred = torch.squeeze(pred, 1)
        if self.classification:
            loss = nn.BCELoss(
                weight=torch.tensor([self.class_weight[0] if l == 0 else self.class_weight[1] for l in labels]))
        else:
            loss = nn.MSELoss()
        self.log("val_loss", loss(pred, labels), prog_bar=True)
        return loss(pred, labels)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lrdecay = torch.optim.lr_scheduler.ExponentialLR(optim, self.hyperparameters.get("lr_decay", 0.9))
        return [optim], [lrdecay]


class ConvClassifier(ClassificationModel):

    def __init__(self, model_config, random_state):
        super(ConvClassifier, self).__init__(model_config, random_state)
        torch.manual_seed(random_state)
        self.model = ConvNet(**model_config["model_config"])

    def train(self, x, y, x_val=None, y_val=None):
        assert x_val is not None and y_val is not None, "Validation set not given!"
        x = np.transpose(x, (0, 2, 1))
        x_val = np.transpose(x_val, (0, 2, 1))
        x = torch.tensor(x).float()
        y = torch.tensor(y)
        x_val = torch.tensor(x_val).float()
        y_val = torch.tensor(y_val)
        train_set = TensorDataset(x, y)
        val_set = TensorDataset(x_val, y_val)
        cpus = cpu_count()
        if self.model_config.get("sampling") == "balanced":
            train_loader = DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set),
                                      batch_size=self.model_config["batch_size"], num_workers=cpus)
            val_loader = DataLoader(val_set, sampler=ImbalancedDatasetSampler(val_set),
                                    batch_size=self.model_config["batch_size"], num_workers=cpus)
        else:
            train_loader = DataLoader(train_set, batch_size=self.model_config["batch_size"], num_workers=cpus)
            val_loader = DataLoader(val_set, batch_size=self.model_config["batch_size"], num_workers=cpus)
        trainer = pl.Trainer(**self.model_config["trainer_config"])
        trainer.fit(self.model, train_loader, val_loader)
        if self.model_config.get("sampling") == "balanced":
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model.forward(x_val)
            self.model.train()
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_val_pred.numpy(), y_val.numpy())
            self.ratio = (y_val.sum() + y.sum()) / (len(y) + len(y_val))
            self.calibrator = LogisticRegression(n_jobs=-1,
                                                 class_weight={1: 1 / torch.sqrt(self.ratio),
                                                               0: 1 / torch.sqrt(1 - self.ratio)})
            self.calibrator.fit(y_val_pred.numpy(), y_val.numpy())
        elif self.model_config.get("sampling") == "balanced_best":
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model.forward(x_val)
            self.model.train()
            self.th, _ = find_best_threshold(y_val.numpy(), y_val_pred.numpy(), 100)

    def predict(self, x):
        self.model.eval()
        x = np.transpose(x, (0, 2, 1))
        x = torch.tensor(x).float()
        with torch.no_grad():
            res = self.model.forward(x)
        self.model.train()
        res = res.numpy()
        if self.model_config.get("sampling") == "balanced":
            res = self.calibrator.predict_proba(res)[:, 1]
            res = res > ((np.sqrt(self.ratio) + self.ratio) / 2).numpy()
        elif self.model_config.get("sampling") == "balanced_best":
            res = res > self.th
        else:
            res = np.round(res)
        return res

    def get_parameters(self):
        return self.model.hparams

    def is_parallelizable(self):
        return True

    def is_multivariate(self):
        return True

    def predict_proba(self, x):
        self.model.eval()
        x = torch.tensor(x).float()
        with torch.no_grad():
            res = self.model.forward(x)
        self.model.train()
        res = res.numpy()
        if self.model_config.get("sampling") == "balanced":
            res = self.calibrator.predict_proba(res)[:, 1]
        return res

    @staticmethod
    def input_format():
        return "time"
