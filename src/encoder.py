import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import math
from src.constants import BATCH_SIZE, FEEDFORWARD_DIM, DROPOUT, D_MODEL, N_HEADS, N_LAYERS, WINDOW_SIZE, PRETRAIN_EPOCHS, PATIENCE, FINETUNING_EPOCHS

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition ---
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, feedforward_dim, dropout=0.2):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model)
        )
        self.batch_norm1 = nn.BatchNorm1d(d_model)
        self.batch_norm2 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.batch_norm1(x.transpose(1, 2)).transpose(1, 2)
        ffn_output = self.feedforward(x)
        x = x + self.dropout2(ffn_output)
        x = self.batch_norm2(x.transpose(1, 2)).transpose(1, 2)
        return x

def _create_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)  # Shape: [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.unsqueeze(0)  # Add batch dimension, Shape: [1, seq_len, d_model]

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=128, n_heads=4, n_layers=2, feedforward_dim=256, dropout=0.1, learnable_positional_encoding: bool=True):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        if learnable_positional_encoding: self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        else: self.positional_encoding = _create_positional_encoding(seq_len, d_model)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, n_heads, feedforward_dim, dropout)
            for _ in range(n_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        x = self.input_embedding(x) + self.positional_encoding
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)

class MaskedPredictionModel(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=128, n_heads=4, n_layers=2, feedforward_dim=256, dropout=0.1, learnable_positional_encoding: bool=True):
        super(MaskedPredictionModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        if learnable_positional_encoding: self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        else: self.positional_encoding = _create_positional_encoding(seq_len, d_model)

        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, n_heads, feedforward_dim, dropout)
            for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_embedding(x) + self.positional_encoding
        for layer in self.encoder_layers:
            x = layer(x)
        return self.output_layer(x)

# --- evaluate and, train and predict ---
def _evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    path: str,
    learnable_positional_encoding: bool,
    pretrain_epochs: int= PRETRAIN_EPOCHS,
):

    n_features = X_train.shape[2]
    window_size = X_train.shape[1]

    # --- Convert numpy arrays to PyTorch tensors ---
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


    train_dataset = TensorDataset(X_tensor, Y_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Masked Prediction Training ---
    def create_masked_input(x, mask_prob=0.15):
        masked_x = x.clone()
        mask = torch.rand_like(x) < mask_prob
        masked_x[mask] = 0.0
        return masked_x, mask

    def train_masked_prediction(model, dataloader, optimizer, criterion, device, mask_prob=0.15):
        model.train()
        total_loss = 0
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            masked_x, mask = create_masked_input(X_batch, mask_prob)
            optimizer.zero_grad()
            predictions = model(masked_x)
            loss = criterion(predictions[mask], X_batch[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    # --- Pretraining ---
    masked_model = MaskedPredictionModel(
        input_dim=n_features,
        seq_len=window_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        learnable_positional_encoding=learnable_positional_encoding,
    ).to(_device)

    optimizer = torch.optim.Adam(masked_model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(pretrain_epochs):
        train_loss = train_masked_prediction(masked_model, train_loader, optimizer, criterion, _device)
        print(f"Pretraining Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Train Loss: {train_loss:.4f}")

    # --- Transfer Weights for Fine-Tuning ---
    regressor_model = TransformerRegressor(
        input_dim=n_features, seq_len=window_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, feedforward_dim=FEEDFORWARD_DIM, dropout=DROPOUT,
        learnable_positional_encoding=learnable_positional_encoding
    ).to(_device)
    if PRETRAIN_EPOCHS > 0:
        regressor_model.load_state_dict(masked_model.state_dict(), strict=False)

    # --- Fine-Tuning ---

    # --- Training and Evaluation Functions ---
    def train_one_epoch(model, dataloader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    # --- Fine-Tuning with Early Stopping ---
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, regressor_model.parameters()), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(FINETUNING_EPOCHS):
        train_loss = train_one_epoch(regressor_model, train_loader, optimizer, criterion, _device)
        val_loss = _evaluate(regressor_model, val_loader, criterion, _device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(regressor_model.state_dict(), path)
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter

        print(f"Fine-Tuning Epoch {epoch+1}/{FINETUNING_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    # --- Load Best Model ---
    regressor_model.load_state_dict(torch.load(path))
    final_val_loss = _evaluate(regressor_model, val_loader, criterion, _device)
    print(f"Final Validation Loss: {final_val_loss:.4f}")

def predict(X_test: np.ndarray, path: str, learnable_positional_encoding: bool):
    
    # --- Load Pretrained Model ---
    regressor_model = TransformerRegressor(input_dim=int(X_test.shape[2]),  # Feature count after drops
                                        seq_len=X_test.shape[1], d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                                        feedforward_dim=FEEDFORWARD_DIM, dropout=DROPOUT, learnable_positional_encoding=learnable_positional_encoding).to(_device)
    regressor_model.load_state_dict(torch.load(path))
    regressor_model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    # --- Get model predictions ---
    with torch.no_grad():
        predictions = regressor_model(X_tensor.to(_device)).cpu().numpy()
    
    return predictions
