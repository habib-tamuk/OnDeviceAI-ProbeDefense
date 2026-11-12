# -*- coding: utf-8 -*-
"""
Defending On-Device AI from Black-Box Probing Attacks with an Autoencoder
IEEE ScSETech 2026 tutorial - Case A: CIFAR-10 (ResNet-20 outputs)

PyTorch version of the "CIFAR-10 AE.py" script used for our ScSETech 2026 paper.
Same windows (600 x 15), same autoencoder, same threshold rule (mean + 2.5 * std,
flag if MSE or MAE is too high). Changes for the tutorial:
  * PyTorch instead of TensorFlow/Keras
  * thresholds are computed on VALIDATION windows (never on the test windows)
  * fixed random seeds, relative paths, clear error messages
  * the extra "stability" rule from the original script is left out (it is not
    part of the paper's method)

HOW TO RUN (Spyder or VS Code)
  The file is split into cells that start with "#%%".
  Click inside a cell and press  Ctrl+Enter        (run this cell)
                            or   Shift+Enter       (run this cell and go to the next)
  Run the cells IN ORDER, from Block 1 to Block 10.
"""

#%% ===================================================================
# BLOCK 1 - Libraries, settings and folders
# =====================================================================
import os
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# >>> CHANGE THIS LINE: YOUR DATA DIRECTORY <<<
# The folder that contains Benign.csv, HSJA.csv, NES.csv, QEBA.csv,
# Square.csv, SURFREE.csv and Boundary.csv (the "CIFAR-10" folder of the
# Kaggle download).
# Windows example: r"C:\Users\YourName\Downloads\oars-dataset\Dataset\CIFAR-10"
# macOS example:   "/Users/yourname/Downloads/oars-dataset/Dataset/CIFAR-10"
DATA_DIR = r"C:\Users\YourName\Downloads\oars-dataset\Dataset\CIFAR-10"   # <-- your data directory

#DATA_DIR = r"C:\Users\USER\Downloads\archive\Dataset\CIFAR-10"
# ---------------------------------------------------------------------

# >>> YOUR OUTPUT DIRECTORY <<<  (results, figures and the trained model go here)
OUTPUT_DIR = os.path.join(os.getcwd(), "tutorial_outputs")   # <-- change if you like
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "benign":   "Benign.csv",
    "hsja":     "HSJA.csv",
    "nes":      "NES.csv",
    "qeba":     "QEBA.csv",
    "square":   "Square.csv",
    "surfree":  "SURFREE.csv",
    "boundary": "Boundary.csv",
}
ATTACKS = ["hsja", "qeba", "nes", "square", "surfree", "boundary"]

# Settings from the paper (change them later as an exercise)
N_ROWS      = 14000   # read only the first 14,000 rows of each file
WINDOW_SIZE = 600     # queries per window
STRIDE      = 60      # the next window starts 60 queries later
CHUNK_SIZE  = 100     # 600 / 100 = 6 segments per window
K           = 2.5     # threshold = mean + K * std
SEED        = 42

def set_seed(seed=SEED):
    """Make every run give the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()
device = torch.device("cpu")   # the tutorial runs on a normal laptop CPU

print("Python libraries loaded.")
print("PyTorch version:", torch.__version__)
print("Data directory  :", DATA_DIR)
print("Output directory:", OUTPUT_DIR)

# Check that the data files are where we expect them
missing = [name for name in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, name))]
if missing:
    print("\nWARNING - these files were not found in DATA_DIR:", missing)
    print("Fix DATA_DIR above (it must point to the CIFAR-10 folder).")
    print("Missing attack files are skipped later; Benign.csv is required.")
else:
    print("All 7 data files found.")

#%% ===================================================================
# BLOCK 2 - Load the data and look at it
# =====================================================================
COLUMNS = ["SortedList_First_Element", "SortedList_Second_Element", "Margin Loss"]

def load_data_local(file_name, n_rows=N_ROWS):
    """Read the 3 columns we need from one CSV file (first n_rows rows only)."""
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Check DATA_DIR in Block 1.")
    df = pd.read_csv(path, usecols=COLUMNS, nrows=n_rows)
    return df

real_df = load_data_local(FILES["benign"])
print("Benign data shape (rows, columns):", real_df.shape)
print(real_df.head())

# Each row is ONE query to the model:
#   top-1 probability, top-2 probability, margin = top-1 - top-2
gap = (real_df[COLUMNS[0]] - real_df[COLUMNS[1]] - real_df[COLUMNS[2]]).abs().max()
print(f"\nCheck: margin equals top-1 minus top-2 (largest difference = {gap:.1e})")

# Plot the first 600 benign queries next to the first 600 queries of one attack
attack_preview = load_data_local(FILES["square"], n_rows=600) if os.path.exists(
    os.path.join(DATA_DIR, FILES["square"])) else None
fig, axes = plt.subplots(1, 2 if attack_preview is not None else 1, figsize=(12, 3.5), sharey=True)
axes = np.atleast_1d(axes)
for ax, (title, d) in zip(axes, [("Benign", real_df.iloc[:600]), ("Square attack", attack_preview)]):
    if d is None:
        continue
    ax.plot(d[COLUMNS[0]].values, label="Top-1 probability")
    ax.plot(d[COLUMNS[1]].values, "--", label="Top-2 probability")
    ax.set_title(title); ax.set_xlabel("Query index")
axes[0].set_ylabel("Probability"); axes[0].legend()
plt.tight_layout()

# Save the figure to the output folder (set OUTPUT_DIR in Block 1)
fig.savefig(os.path.join(OUTPUT_DIR, "benign_vs_attack_traces.png"), dpi=200, bbox_inches="tight")
print("Figure saved to", os.path.join(OUTPUT_DIR, "benign_vs_attack_traces.png"))

plt.show()
#%% ===================================================================
# BLOCK 3 - Split the benign queries: 60% train, 20% validation, 20% test
# =====================================================================
# We split the QUERIES first and only then cut windows, so no query can be in
# two subsets (no data leakage).
train_df, temp_df = train_test_split(real_df, test_size=0.4, random_state=SEED)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED)
train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))

print("Benign queries -> train:", len(train_df), " validation:", len(val_df), " test:", len(test_df))

#%% ===================================================================
# BLOCK 4 - Turn the query stream into 600 x 15 windows
# =====================================================================
def extract_windows(df, window_size=WINDOW_SIZE, stride=STRIDE, chunk_size=CHUNK_SIZE):
    """
    For every window of 600 consecutive queries build a 600 x 15 table:
      columns 1-3  : top-1, top-2, margin (for each of the 600 queries)
      columns 4-9  : mean of the margin in each of the six 100-query segments
      columns 10-15: variance of the margin in each segment
    Columns 4-15 hold one number each, repeated down all 600 rows.
    """
    top    = df[COLUMNS[0]].to_numpy()
    second = df[COLUMNS[1]].to_numpy()
    margin = df[COLUMNS[2]].to_numpy()
    n_chunks = window_size // chunk_size
    windows = []
    for i in range(0, len(top) - window_size + 1, stride):
        w1, w2, w3 = top[i:i+window_size], second[i:i+window_size], margin[i:i+window_size]
        means = [np.full(window_size, w3[j*chunk_size:(j+1)*chunk_size].mean()) for j in range(n_chunks)]
        varis = [np.full(window_size, w3[j*chunk_size:(j+1)*chunk_size].var())  for j in range(n_chunks)]
        windows.append(np.column_stack([w1, w2, w3] + means + varis))
    return np.array(windows, dtype=np.float32)

train_windows = extract_windows(train_df)
val_windows   = extract_windows(val_df)
test_windows  = extract_windows(test_df)

print("Train windows:", train_windows.shape)   # (number of windows, 600, 15)
print("Val windows:  ", val_windows.shape)
print("Test windows: ", test_windows.shape)

#%% ===================================================================
# BLOCK 5 - The convolutional autoencoder (PyTorch)
# =====================================================================
class ConvAE(nn.Module):
    """
    Encoder: 600x15 -> 300x64 -> 150x32 -> 75x16 -> bottleneck 75x8
    Decoder: 75x8 -> 150x16 -> 300x32 -> 600x64 -> 600x15
    Same layers as the Keras model in our papers.
    """
    def __init__(self, n_features=15):
        super().__init__()
        def conv_block(c_in, c_out, k):
            # Conv -> ReLU -> BatchNorm  (BatchNorm settings match Keras defaults)
            return [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.ReLU(),
                    nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01)]
        def up_block(c_in, c_out, k):
            # Transposed conv doubles the length (75 -> 150 -> 300 -> 600)
            return [nn.ConvTranspose1d(c_in, c_out, k, stride=2, padding=k // 2, output_padding=1),
                    nn.ReLU(), nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01)]
        self.encoder = nn.Sequential(
            *conv_block(n_features, 64, 5), nn.MaxPool1d(2),
            *conv_block(64, 32, 7),         nn.MaxPool1d(2),
            *conv_block(32, 16, 9),         nn.MaxPool1d(2),
            *conv_block(16, 8, 11),                        # bottleneck 75 x 8
        )
        self.decoder = nn.Sequential(
            *up_block(8, 16, 9),
            *up_block(16, 32, 7),
            *up_block(32, 64, 5),
            nn.Conv1d(64, n_features, 3, padding=1),       # linear output
        )

    def forward(self, x):              # x: (batch, 600, 15)
        x = x.transpose(1, 2)          # PyTorch wants (batch, channels=15, length=600)
        x = self.decoder(self.encoder(x))
        return x.transpose(1, 2)       # back to (batch, 600, 15)

model = ConvAE().to(device)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"\nTrainable parameters: {n_trainable:,}   (paper's Keras model: 43,719)")

# Quick check: the output must have the same shape as the input
with torch.no_grad():
    model.eval()
    print("Output shape for one window:", tuple(model(torch.from_numpy(train_windows[:1])).shape))

#%% ===================================================================
# BLOCK 6 - Train the autoencoder on BENIGN windows only
# =====================================================================
LOAD_SAVED_MODEL = False   # set to True to skip training and load the saved model
MODEL_PATH = os.path.join(OUTPUT_DIR, "cifar10_conv_ae.pt")   # <-- your model file
EPOCHS, BATCH_SIZE, PATIENCE = 100, 32, 50

if LOAD_SAVED_MODEL:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    history = None
    print("Loaded saved model from", MODEL_PATH)
else:
    set_seed()
    model = ConvAE().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-4)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_windows)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_tensor = torch.from_numpy(val_windows)

    history = {"loss": [], "val_loss": []}
    best_val, best_state, waited = float("inf"), None, 0
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for (xb,) in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), xb)          # input = target: learn to copy
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_tensor), val_tensor).item()
        history["loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:                   # early stopping: keep the best model
            best_val, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  train loss {history['loss'][-1]:.6f}  val loss {val_loss:.6f}")
        if waited >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training took {time.time() - start:.0f} s. Best validation loss {best_val:.6f}")
    print("Model saved to", MODEL_PATH)

#%% ===================================================================
# BLOCK 7 - Plot the training curve
# =====================================================================
if history is not None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["loss"], "--", label="Train loss")
    plt.plot(history["val_loss"], "--", label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.yscale("log"); plt.legend()
    plt.title("Autoencoder training on benign windows")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=200)
    plt.show()
else:
    print("Model was loaded, so there is no new training curve.")

#%% ===================================================================
# BLOCK 8 - Reconstruction errors and the alarm thresholds
# =====================================================================
def reconstruction_errors(windows):
    """Return MSE and MAE for every window."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(windows)
        x_hat = model(x)
    mse = ((x - x_hat) ** 2).mean(dim=(1, 2)).numpy()
    mae = (x - x_hat).abs().mean(dim=(1, 2)).numpy()
    return {"mse": mse, "mae": mae}

# Thresholds come from the VALIDATION windows only (never the test windows)
val_errors = reconstruction_errors(val_windows)
thresholds = {m: float(v.mean() + K * v.std()) for m, v in val_errors.items()}
print("Thresholds (mean + 2.5 * std of validation errors):")
for m, t in thresholds.items():
    print(f"  {m.upper()}: {t:.6f}")
with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as f:
    json.dump(thresholds, f, indent=2)

def predict(errors):
    """Flag a window (1) if MSE OR MAE is above its threshold."""
    return ((errors["mse"] > thresholds["mse"]) | (errors["mae"] > thresholds["mae"])).astype(int)

test_errors = reconstruction_errors(test_windows)
false_alarms = predict(test_errors).sum()
print(f"\nBenign test windows flagged by mistake: {false_alarms} of {len(test_windows)}")

#%% ===================================================================
# BLOCK 9 - Test against each probing attack
# =====================================================================
summary = []
for atk in ATTACKS:
    try:
        attack_windows = extract_windows(load_data_local(FILES[atk]))
    except FileNotFoundError as e:
        print(f"Skipping {atk.upper()}: {e}")
        continue
    attack_errors = reconstruction_errors(attack_windows)

    y_true = np.concatenate([np.zeros(len(test_windows)), np.ones(len(attack_windows))])
    y_pred = np.concatenate([predict(test_errors), predict(attack_errors)])
    print(f"\n===== {atk.upper()} vs benign =====")
    print(classification_report(y_true, y_pred, target_names=["Benign", atk.upper()], digits=2))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    summary.append({"attack": atk.upper(), "attack windows": len(attack_windows),
                    "detected": int(tp), "missed": int(fn),
                    "benign false alarms": int(fp), "benign windows": len(test_windows)})

    # Confusion matrix and error histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Benign", atk.upper()], yticklabels=["Benign", atk.upper()])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    for ax, m in zip(axes[1:], ["mse", "mae"]):
        ax.hist(test_errors[m], bins=30, alpha=0.6, label="Benign test")
        ax.hist(attack_errors[m], bins=30, alpha=0.6, label=atk.upper())
        ax.axvline(thresholds[m], color="black", linestyle="--", label="Threshold")
        ax.set_xscale("log"); ax.set_title(m.upper() + " per window"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{atk}_results.png"), dpi=200)
    plt.show()

summary_df = pd.DataFrame(summary)
print("\nSUMMARY")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

#%% ===================================================================
# BLOCK 10 - Live demo: hide attacks inside a benign stream
# =====================================================================
# Benign test windows are repeated (with tiny noise) to make a long stream,
# then short runs of attack windows are inserted. This is an illustration of
# how the detector behaves over time, not an independent test.
set_seed()
benign_stream = np.tile(test_windows, (6, 1, 1))
benign_stream = benign_stream + np.random.normal(0, 0.001, benign_stream.shape).astype(np.float32)

pieces, labels, inserted = [benign_stream[:30]], [np.zeros(30, int)], []
position = 30
for atk in ATTACKS:
    try:
        attack_windows = extract_windows(load_data_local(FILES[atk]))
    except FileNotFoundError:
        continue
    length = random.choice([5, 10, 20])
    start = random.randint(0, len(attack_windows) - length)
    pieces += [attack_windows[start:start+length], benign_stream[position:position+15]]
    labels += [np.ones(length, int), np.zeros(15, int)]
    inserted.append((atk.upper(), sum(len(p) for p in pieces[:-2]), length))
    position += 15
stream = np.concatenate(pieces).astype(np.float32)
truth = np.concatenate(labels)
flags = predict(reconstruction_errors(stream))

print(f"{'Attack':10s} | {'Windows':>8s} | {'Flagged':>8s}")
for name, begin, length in inserted:
    print(f"{name:10s} | {begin:4d}-{begin+length:<4d}| {flags[begin:begin+length].sum():3d} of {length}")

plt.figure(figsize=(14, 3))
plt.plot(truth, "k--", label="Ground truth (1 = attack)")
plt.plot(flags, "r", alpha=0.7, label="Detector flag")
plt.xlabel("Window index in the stream"); plt.ylabel("Flag"); plt.legend(loc="upper right")
plt.title("Attacks hidden in a benign stream")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stream_demo.png"), dpi=200)
plt.show()
print(classification_report(truth, flags, target_names=["Benign", "Attack"], digits=2))
