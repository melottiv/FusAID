import torch
import torch.nn as nn
import copy
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, config, y_train=None,noise=0.01):
        """
        model      : il tuo modello PyTorch
        optimizer  : ottimizzatore
        config     : config contenente device, epochs, early_stopping_patience, ecc.
        y_train    : array o tensor con le labels del training set (0/1), serve per calcolare pos_weight
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.noise=noise


        # ======== Aggiunta class weights ========
        if y_train is not None:
            # Calcolo peso della classe positiva
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            N = len(y_train_tensor)
            N_pos = (y_train_tensor == 1).sum()
            N_neg = N - N_pos
            pos_weight = N_neg / N_pos
            print(f"[INFO] Using pos_weight={pos_weight:.3f} for BCE loss")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Early stopping
        self.best_loss = float('inf')
        self.best_model_state = None
        self.no_improve_epochs = 0

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for x, y, _ in loader:
            x = x.to(self.device)
            x = x + torch.randn_like(x) * self.noise    
            y = y.to(self.device).float().view(-1, 1)  # garantiamo shape (B,1)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def train_model(trainer, train_loader, val_loader, config):

        best_loss = float("inf")
        best_state = None

        patience = getattr(config, "patience", 10)
        min_delta = getattr(config, "min_delta", 1e-4)

        counter = 0

        for epoch in range(config.epochs):

            train_loss = trainer.train_epoch(train_loader)
            val_loss, _, _ ,_= trainer.evaluate(val_loader)

            print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = trainer.model.state_dict()
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        trainer.model.load_state_dict(best_state)


    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_logits = []
        all_labels = []
        all_ids = []

        with torch.no_grad():

            for x, y, rid in loader:
                logits = self.model(x.to(self.device))

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())
                all_ids.extend(rid)

        return (
            total_loss / len(loader),
            torch.cat(all_logits),
            torch.cat(all_labels),
            np.array(all_ids)
        )

    def train(self, train_loader, val_loader):
        for epoch in range(self.config.epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.no_improve_epochs = 0
                print("Validation improved, saving model...")
            else:
                self.no_improve_epochs += 1
                print(f"No improvement for {self.no_improve_epochs} epochs.")

            if self.no_improve_epochs >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)