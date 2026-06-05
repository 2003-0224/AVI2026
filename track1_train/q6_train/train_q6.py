import os
import sys
import time
import copy
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import Dataset, DataLoader

from model import TextCenteredCrossModalAttentionRegressor


# =============================
# 1. Arguments
# =============================
if len(sys.argv) < 7:
    print("Usage: python train.py <BASE_DIR> <ROOT_OUTPUT_DIR> <MODEL_SAVE_DIR> <TRAIN_CSV> <VAL_CSV> <QTYPE>")
    sys.exit(1)

BASE_DIR = sys.argv[1]
ROOT_OUTPUT_DIR = sys.argv[2]
BASE_MODEL_SAVE_DIR = sys.argv[3]
TRAIN_CSV_PATH = sys.argv[4]
VAL_CSV_PATH = sys.argv[5]
CHOSEN_QTYPE = sys.argv[6]

assert CHOSEN_QTYPE == "q6", "This script is configured for q6 / Conscientiousness."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 10
SEEDS = [42, 1023, 2026]

EMBED_DIM = 1536

QUESTION_CONFIGS = {
    "q6": {"label_col": "C_self", "trait": "Conscientiousness"},
}

MODALITY_MAP = {
    "t": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_t"),
    "a": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_a"),
    "v": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_v"),
}

MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, CHOSEN_QTYPE)

GRID_SEARCH_PARAMS = {
    "lr": [1e-4, 5e-5, 1e-5],
    "weight_decay": [0, 1e-3, 1e-2, 5e-2],
    "dropout_rate": [0.0, 0.1, 0.3, 0.5],
    "hidden_dims": [(128,), (256,), (512,), (256, 64), (128, 32), (512, 256, 64)],
    "attn_dim": [256, 512, 768],
    "num_heads": [4, 8],
}


# =============================
# 2. Utilities
# =============================
def setup_logging(out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"log_q6_cross_attention_seed_{seed}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(f"q6_cross_attention_{seed}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_bins(y, n_bins=5):
    return pd.cut(y, bins=n_bins, labels=False, duplicates="drop")


# =============================
# 3. Dataset
# =============================
class CrossModalDataset(Dataset):
    def __init__(self, df, q_type, label_col=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.q_type = q_type
        self.label_col = label_col
        self.is_test = is_test

    def _load_emb(self, mod_key, sid):
        base_dir = MODALITY_MAP[mod_key]
        fname = f"{sid}_{self.q_type}.npz"

        for sub in ["train", "val", "test", ""]:
            path = os.path.join(base_dir, sub, fname)
            if os.path.exists(path):
                with np.load(path) as d:
                    return d["embedding"].astype(np.float32)

        raise FileNotFoundError(f"Missing {mod_key} embedding for sample {sid}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row["id"])

        text = torch.from_numpy(self._load_emb("t", sid)).float()
        audio = torch.from_numpy(self._load_emb("a", sid)).float()
        video = torch.from_numpy(self._load_emb("v", sid)).float()

        if self.is_test or self.label_col is None:
            label = torch.tensor(0.0).float()
        else:
            label = torch.tensor(row["y_trans"]).float()

        return sid, text, audio, video, label


# =============================
# 4. Train / Eval
# =============================
def train_eval_flow(model, loaders, transformer, lr, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_trans_mse = float("inf")
    best_orig_mse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for _, xt, xa, xv, yb in loaders["train"]:
            xt = xt.to(DEVICE)
            xa = xa.to(DEVICE)
            xv = xv.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xt, xa, xv)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for _, xt, xa, xv, yb in loaders["val"]:
                xt = xt.to(DEVICE)
                xa = xa.to(DEVICE)
                xv = xv.to(DEVICE)

                pred = model(xt, xa, xv).cpu().numpy()
                preds.append(pred)
                labels.append(yb.numpy())

        pred_trans = np.concatenate(preds)
        label_trans = np.concatenate(labels)

        trans_mse = mean_squared_error(label_trans, pred_trans)

        if trans_mse < best_trans_mse:
            pred_orig = transformer.inverse_transform(pred_trans.reshape(-1, 1)).reshape(-1)
            label_orig = transformer.inverse_transform(label_trans.reshape(-1, 1)).reshape(-1)

            best_trans_mse = trans_mse
            best_orig_mse = mean_squared_error(label_orig, pred_orig)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_trans_mse, best_orig_mse, best_state


def build_model(hparams):
    return TextCenteredCrossModalAttentionRegressor(
        embed_dim=EMBED_DIM,
        attn_dim=hparams["attn_dim"],
        num_heads=hparams["num_heads"],
        hidden_dims=hparams["hidden_dims"],
        dropout_rate=hparams["dropout_rate"],
    ).to(DEVICE)


# =============================
# 5. Main
# =============================
def main():
    config = QUESTION_CONFIGS[CHOSEN_QTYPE]

    train_data = pd.read_csv(TRAIN_CSV_PATH)
    val_data = pd.read_csv(VAL_CSV_PATH)
    all_data = pd.concat([train_data, val_data], ignore_index=True)

    label_col = config["label_col"]

    # Yeo-Johnson transformation on merged train+val labels.
    # During CV, the transformer is fitted only on each fold's training labels.
    all_data["bins"] = make_bins(all_data[label_col], n_bins=5)

    param_grid = list(product(
        GRID_SEARCH_PARAMS["lr"],
        GRID_SEARCH_PARAMS["weight_decay"],
        GRID_SEARCH_PARAMS["dropout_rate"],
        GRID_SEARCH_PARAMS["hidden_dims"],
        GRID_SEARCH_PARAMS["attn_dim"],
        GRID_SEARCH_PARAMS["num_heads"],
    ))

    for seed in SEEDS:
        set_seed(seed)

        out_dir = os.path.join(ROOT_OUTPUT_DIR, f"q6_{config['trait']}_cross_attention_seed_{seed}")
        model_dir = os.path.join(MODEL_SAVE_DIR, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        logger = setup_logging(out_dir, seed)
        logger.info(f"Start q6 Cross-Modal Attention training | Seed={seed}")

        # =============================
        # Grid search with 5-fold CV
        # =============================
        best_cv_mse = float("inf")
        best_hparams = None

        for lr, wd, dr, hds, attn_dim, num_heads in tqdm(param_grid, desc=f"Grid Search Seed {seed}"):
            if attn_dim % num_heads != 0:
                continue

            hparams = {
                "lr": lr,
                "weight_decay": wd,
                "dropout_rate": dr,
                "hidden_dims": hds,
                "attn_dim": attn_dim,
                "num_heads": num_heads,
            }

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_mses = []

            for fold, (tr_idx, va_idx) in enumerate(skf.split(all_data, all_data["bins"])):
                fold_train = all_data.iloc[tr_idx].copy()
                fold_val = all_data.iloc[va_idx].copy()

                transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                fold_train["y_trans"] = transformer.fit_transform(fold_train[[label_col]]).reshape(-1)
                fold_val["y_trans"] = transformer.transform(fold_val[[label_col]]).reshape(-1)

                loaders = {
                    "train": DataLoader(
                        CrossModalDataset(fold_train, CHOSEN_QTYPE, label_col),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                    ),
                    "val": DataLoader(
                        CrossModalDataset(fold_val, CHOSEN_QTYPE, label_col),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                    ),
                }

                model = build_model(hparams)
                _, orig_mse, _ = train_eval_flow(model, loaders, transformer, lr, wd)
                fold_mses.append(orig_mse)

            avg_mse = float(np.mean(fold_mses))

            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_hparams = hparams

        logger.info(f"Best CV MSE: {best_cv_mse:.6f}")
        logger.info(f"Best Params: {best_hparams}")

        # =============================
        # Train and save all 5 folds for ensemble
        # =============================
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_states = []
        fold_mses = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(all_data, all_data["bins"])):
            fold_train = all_data.iloc[tr_idx].copy()
            fold_val = all_data.iloc[va_idx].copy()

            transformer = PowerTransformer(method="yeo-johnson", standardize=True)
            fold_train["y_trans"] = transformer.fit_transform(fold_train[[label_col]]).reshape(-1)
            fold_val["y_trans"] = transformer.transform(fold_val[[label_col]]).reshape(-1)

            loaders = {
                "train": DataLoader(
                    CrossModalDataset(fold_train, CHOSEN_QTYPE, label_col),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                ),
                "val": DataLoader(
                    CrossModalDataset(fold_val, CHOSEN_QTYPE, label_col),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                ),
            }

            model = build_model(best_hparams)
            _, orig_mse, best_state = train_eval_flow(
                model,
                loaders,
                transformer,
                best_hparams["lr"],
                best_hparams["weight_decay"],
            )

            fold_mses.append(orig_mse)
            fold_states.append(best_state)

            save_path = os.path.join(model_dir, f"q6_cross_attention_fold_{fold}.pth")
            torch.save({
                "state_dict": best_state,
                "transformer": transformer,
                "config": config,
                "hparams": best_hparams,
                "fold": fold,
                "fold_mse": orig_mse,
                "model_type": "text_centered_cross_modal_attention",
            }, save_path)

            logger.info(f"Saved Fold {fold} | MSE={orig_mse:.6f} | Path={save_path}")

        logger.info(f"Final 5-Fold Average MSE: {np.mean(fold_mses):.6f}")
        logger.info(f"Fold MSEs: {fold_mses}")


if __name__ == "__main__":
    main()
