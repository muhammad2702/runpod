#FOCAL LOSS



import os
import pandas as pd
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.init import xavier_uniform_ as xavier_uniform
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score  # If you want to use balanced accuracy
from torch.nn.init import xavier_uniform_
# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape [num_classes], or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Gather the log probability of the target class only
        batch_indices = torch.arange(len(targets), dtype=torch.long, device=targets.device)
        pt = probs[batch_indices, targets]       # shape: [batch_size]
        log_pt = log_probs[batch_indices, targets]  # shape: [batch_size]

        # Apply alpha if provided
        if self.alpha is not None:
            # alpha is [num_classes], select alpha for each target
            at = self.alpha[targets]  # shape: [batch_size]
        else:
            at = 1.0

        # Compute focal loss
        # FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
        focal_loss = -at * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ShortTermTransformerModel(nn.Module):
    def __init__(self, num_features, num_cryptos, d_model=64, nhead=2, num_encoder_layers=1,
                 dim_feedforward=64, dropout=0.1, num_classes=7, max_seq_length=50):
        super(ShortTermTransformerModel, self).__init__()
        self.d_model = d_model
        self.crypto_embedding = nn.Embedding(num_cryptos, d_model)
        
        self.input_linear = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.PReLU(),
            nn.LayerNorm(d_model)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.percent_change_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.leg_direction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )

        self.initialize_weights()

    def forward(self, src, crypto_id):
        # Transform input features and add crypto embedding
        src = self.input_linear(src)
        crypto_emb = self.crypto_embedding(crypto_id).unsqueeze(1)
        src = src + crypto_emb

        # Pass through Transformer
        memory = self.transformer_encoder(src)
        # Global average pooling over the sequence dimension
        features = torch.mean(memory, dim=1)

        # Compute logits
        percent_logits = self.percent_change_head(features)
        leg_logits = self.leg_direction_head(features)

        # Convert logits to probability distributions
        percent_probs = nn.functional.softmax(percent_logits, dim=-1)
        leg_probs = nn.functional.softmax(leg_logits, dim=-1)

        return percent_probs, leg_probs

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)



class CryptoDataset(Dataset):
    def __init__(self, dataframe, feature_cols, window_size=60):
        self.data = dataframe
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.cryptos = sorted(dataframe['crypto'].unique())
        self.crypto_to_id = {crypto: idx for idx, crypto in enumerate(self.cryptos)}
        self.crypto_data = {
            crypto: dataframe[dataframe['crypto'] == crypto].sort_values('t').reset_index(drop=True)
            for crypto in self.cryptos
        }

        self.indices = []
        for crypto in self.cryptos:
            data_length = len(self.crypto_data[crypto])
            if data_length > self.window_size:
                self.indices.extend([(crypto, idx) for idx in range(data_length - self.window_size)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        crypto, seq_start = self.indices[idx]
        data = self.crypto_data[crypto].iloc[seq_start:seq_start + self.window_size]
        features = data[self.feature_cols].values

        target_idx = seq_start + self.window_size
        data_length = len(self.crypto_data[crypto])
        if target_idx >= data_length:
            target_idx = data_length - 1

        percent_change = self.crypto_data[crypto].iloc[target_idx]['percent_change_classification']
        leg_direction = self.crypto_data[crypto].iloc[target_idx]['leg_direction']

        crypto_id = self.crypto_to_id[crypto]

        return (
            (torch.tensor(features, dtype=torch.float32), torch.tensor(crypto_id, dtype=torch.long)),
            (
                torch.tensor(percent_change, dtype=torch.long),
                torch.tensor(leg_direction, dtype=torch.long),
            ),
        )


def train(model, dataloader, criterion_dict, optimizer, scheduler, scaler, device, accumulation_steps=2):
    model.train()
    epoch_losses = {"percent_change": 0, "leg_direction": 0, "total": 0}
    metrics = {"percent_change_acc": 0, "leg_direction_acc": 0}
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for idx, ((inputs, crypto_ids), (percent_targets, leg_targets)) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device)
        crypto_ids = crypto_ids.to(device)
        percent_targets = percent_targets.to(device)
        leg_targets = leg_targets.to(device)
        
        with torch.cuda.amp.autocast():
            percent_out, leg_out = model(inputs, crypto_ids)

            loss_percent = criterion_dict['percent_ce'](percent_out, percent_targets)
            loss_leg = criterion_dict['leg_ce'](leg_out, leg_targets)
            
            total_loss = (0.8 * loss_percent + 0.2 * loss_leg) / accumulation_steps
        
        scaler.scale(total_loss).backward()

        batch_size = inputs.size(0)
        total += batch_size
        
        epoch_losses["percent_change"] += loss_percent.item() * batch_size
        epoch_losses["leg_direction"] += loss_leg.item() * batch_size
        epoch_losses["total"] += (total_loss.item() * batch_size * accumulation_steps)
        
        _, predicted_percent = torch.max(percent_out, 1)
        _, predicted_leg = torch.max(leg_out, 1)
        metrics["percent_change_acc"] += (predicted_percent == percent_targets).sum().item()
        metrics["leg_direction_acc"] += (predicted_leg == leg_targets).sum().item()
        
        if (idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    for key in epoch_losses:
        epoch_losses[key] /= total

    metrics["percent_change_acc"] /= total
    metrics["leg_direction_acc"] /= total

    return epoch_losses, metrics


def validate(model, dataloader, criterion_dict, device):
    model.eval()
    val_losses = {"percent_change": 0, "leg_direction": 0, "total": 0}
    val_metrics = {"percent_change_acc": 0, "leg_direction_acc": 0}
    total = 0

    with torch.no_grad():
        all_percent_preds = []
        all_percent_tgts = []
        all_leg_preds = []
        all_leg_tgts = []
        
        for (inputs, crypto_ids), (percent_targets, leg_targets) in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            crypto_ids = crypto_ids.to(device)
            percent_targets = percent_targets.to(device)
            leg_targets = leg_targets.to(device)

            percent_out, leg_out = model(inputs, crypto_ids)

            loss_percent = criterion_dict['percent_ce'](percent_out, percent_targets)
            loss_leg = criterion_dict['leg_ce'](leg_out, leg_targets)
            total_loss = 0.4 * loss_percent + 0.4 * loss_leg

            batch_size = inputs.size(0)
            total += batch_size

            val_losses["percent_change"] += loss_percent.item() * batch_size
            val_losses["leg_direction"] += loss_leg.item() * batch_size
            val_losses["total"] += total_loss.item() * batch_size

            _, predicted_percent = torch.max(percent_out, 1)
            _, predicted_leg = torch.max(leg_out, 1)
            val_metrics["percent_change_acc"] += (predicted_percent == percent_targets).sum().item()
            val_metrics["leg_direction_acc"] += (predicted_leg == leg_targets).sum().item()

            # For optional balanced accuracy
            all_percent_preds.extend(predicted_percent.cpu().numpy())
            all_percent_tgts.extend(percent_targets.cpu().numpy())

    for key in val_losses:
        val_losses[key] /= total

    val_metrics["percent_change_acc"] /= total
    val_metrics["leg_direction_acc"] /= total

    # Example: Compute balanced accuracy for percent_change_classification (optional)
    # ba_percent = balanced_accuracy_score(all_percent_tgts, all_percent_preds)
    # print("Balanced Accuracy (Percent Change):", ba_percent)

    return val_losses, val_metrics


def compute_class_weights(df, target_col):
    class_counts = df[target_col].value_counts().sort_index()
    total = class_counts.sum()
    weights = [total / (len(class_counts) * c) for c in class_counts]
    return torch.tensor(weights, dtype=torch.float)


def main():
    # Parameters
    batch_size = 48
    epochs = 200
    learning_rate = 5e-4
    window_size = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_data_dir = 'preprocessed_data'
    dfs = []
    for ticker in os.listdir(preprocessed_data_dir):
        ticker_dir = os.path.join(preprocessed_data_dir, ticker)
        if os.path.isdir(ticker_dir):
            for file in os.listdir(ticker_dir):
                if file.endswith('_preprocessed.csv'):
                    filepath = os.path.join(ticker_dir, file)
                    df = pd.read_csv(filepath)
                    df['crypto'] = ticker
                    dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    target_cols = ['percent_change_classification', 'leg_direction']
    feature_cols = [col for col in full_df.columns if col not in target_cols + ['t', 'crypto']]

    # Drop NaN
    full_df.dropna(subset=feature_cols + target_cols, inplace=True)

    # Normalize features
    scaler = StandardScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])

    # Shuffle and split
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(0.7 * len(full_df))
    val_size = int(0.15 * len(full_df))

    train_df = full_df.iloc[:train_size]
    val_df = full_df.iloc[train_size:train_size+val_size]
    test_df = full_df.iloc[train_size+val_size:]

    # Print class distributions
    print("Percent change class distribution:")
    print(train_df['percent_change_classification'].value_counts())
    print("Leg direction class distribution:")
    print(train_df['leg_direction'].value_counts())

    # Compute weights for focal loss alpha
    percent_weights = compute_class_weights(train_df, 'percent_change_classification').to(device)
    leg_weights = compute_class_weights(train_df, 'leg_direction').to(device)

    train_dataset = CryptoDataset(train_df, feature_cols, window_size)
    val_dataset = CryptoDataset(val_df, feature_cols, window_size)
    test_dataset = CryptoDataset(test_df, feature_cols, window_size)

    # Prepare WeightedRandomSampler
    # Get targets for weighted sampler
    targets = []
    for i in range(len(train_dataset)):
        _, (pct, _) = train_dataset[i]
        targets.append(int(pct))
    targets = np.array(targets)
    class_sample_counts = np.bincount(targets)
    weight_for_each_class = 1.0 / class_sample_counts
    samples_weight = torch.from_numpy(weight_for_each_class[targets]).double()


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    num_features = len(feature_cols)
    num_cryptos = full_df['crypto'].nunique()
    num_classes = full_df['percent_change_classification'].nunique()

    model = ShortTermTransformerModel(
        num_features=num_features,
        num_cryptos=num_cryptos,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=128,
        num_classes=num_classes,
        max_seq_length=window_size
    ).to(device)

    # Using FocalLoss
    percent_focal_loss = FocalLoss(alpha=percent_weights, gamma=1.0)
    leg_focal_loss = FocalLoss(alpha=leg_weights, gamma=1.0)

    criterion_dict = {
        'percent_ce': percent_focal_loss,
        'leg_ce': leg_focal_loss
    }

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 15

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_losses, train_metrics = train(model, train_loader, criterion_dict, optimizer, scheduler, scaler, device, accumulation_steps=2)
        print(f"Training - Losses: {train_losses}")
        print(f"Training - Metrics: {train_metrics}")

        val_losses, val_metrics = validate(model, val_loader, criterion_dict, device)
        print(f"Validation - Losses: {val_losses}")
        print(f"Validation - Metrics: {val_metrics}")

        # Update scheduler based on validation total loss
        scheduler.step(val_losses["total"])

        # Early stopping
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter > early_stopping_patience:
                print("Early stopping triggered.")
                break

        gc.collect()

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, 'best_short_term_transformer_model.pth')
        print("Best model saved as best_short_term_transformer_model.pth")
    else:
        print("No improvement during training, model not saved.")

    # Final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_losses, test_metrics = validate(model, test_loader, criterion_dict, device)
    print("\nFinal Test Results:")
    print(f"Test Losses: {test_losses}")
    print(f"Test Metrics: {test_metrics}")


if __name__ == '__main__':
    main()

