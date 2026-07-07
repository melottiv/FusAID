from sklearn.preprocessing import StandardScaler
from src.datasets.fusion_dataset import FusionDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models import build_model
from src.training.trainer import Trainer

class SingleModelPipeline:

    def __init__(
        self,
        config,
        input_df,
        emb_path,
        train_ids,
        val_ids,
        task,
        device
    ):

        self.config = config
        self.input_df = input_df
        self.emb_path = emb_path
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.task = task
        self.device = device

        self.scaler = None
        self.trainer = None

        self.train_loader = None
        self.val_loader = None

    def fit_scaler(self):

        dataset = FusionDataset(
            self.input_df,
            self.emb_path,
            self.train_ids,
            self.task
        )

        X = np.stack([
            dataset[i][0].numpy()
            for i in range(len(dataset))
        ])

        self.scaler = StandardScaler().fit(X)

    def normalize(self, x):

        x = self.scaler.transform(
            x.numpy().reshape(1, -1)
        ).squeeze(0)

        return torch.from_numpy(x).float()

    def build_dataloaders(self):

        train_ds = FusionDataset(
            self.input_df,
            self.emb_path,
            self.train_ids,
            self.task,
            transform=self.normalize
        )

        val_ds = FusionDataset(
            self.input_df,
            self.emb_path,
            self.val_ids,
            self.task,
            transform=self.normalize
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        return train_ds

    def build_trainer(self, train_ds):

        model = build_model(
            self.config,
            input_dim=train_ds[0][0].shape[0]
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr
        )

        y_train = torch.cat([
            y for _, y, _ in self.train_loader
        ]).numpy()

        self.trainer = Trainer(
            model,
            optimizer,
            self.config,
            y_train,
            self.config.noise_std
        )

    def train(self):

        self.fit_scaler()

        train_ds = self.build_dataloaders()

        self.build_trainer(train_ds)

        self.trainer.train_model(
            self.train_loader,
            self.val_loader,
            self.config
        )

    def validation_logits(self):

        return self.trainer.evaluate(
            self.val_loader
        )