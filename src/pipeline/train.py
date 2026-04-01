import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from models.deep_alpha import DeepAlphaModel, FocalLoss

logger = logging.getLogger("QuantEngine.Training")

class AlphaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Trainer:
    def __init__(self, model, lr=0.001, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = FocalLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        accuracy = correct / total
        return total_loss / len(dataloader), accuracy

def run_training_pipeline(df, features, target_col='label', batch_size=1024, epochs=10):
    # Prepare data
    df_clean = df.select(features + [target_col]).drop_nulls()

    X = df_clean.select(features).to_numpy()
    y = df_clean[target_col].to_numpy()

    # Handle constant features
    stds = np.std(X, axis=0)
    valid_features_idx = np.where(stds > 1e-9)[0]
    if len(valid_features_idx) < len(features):
        logger.warning(f"Removing {len(features) - len(valid_features_idx)} constant features.")
        X = X[:, valid_features_idx]
        features = [features[i] for i in valid_features_idx]

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Datasets
    train_ds = AlphaDataset(X_train, y_train)
    val_ds = AlphaDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = DeepAlphaModel(input_dim=len(features))
    trainer = Trainer(model)

    logger.info("Starting training...")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, scaler, features
