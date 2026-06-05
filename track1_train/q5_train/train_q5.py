import os, sys, random, copy, logging, torch
import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from itertools import product, combinations

# =============================
# 1. 从 Shell 脚本接收动态参数
# =============================
if len(sys.argv) < 7:
    print("Usage: python run_experiment.py <BASE_DIR> <ROOT_OUTPUT_DIR> <MODEL_SAVE_DIR> <TRAIN_CSV> <VAL_CSV> <QTYPE>")
    sys.exit(1)

BASE_DIR = sys.argv[1]
ROOT_OUTPUT_DIR = sys.argv[2]
BASE_MODEL_SAVE_DIR = sys.argv[3]
TRAIN_CSV_PATH = sys.argv[4]
VAL_CSV_PATH = sys.argv[5]
CHOSEN_QTYPE = sys.argv[6]

MODALITY_MAP = {
    "t": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_t"),
    "a": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_a"),
    "v": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_v"),
}

MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, CHOSEN_QTYPE)
TEMPLATE_PATH = os.path.join(BASE_DIR, "submission.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 200
PATIENCE = 10
SEEDS = [42, 1023, 2026]

QUESTION_CONFIGS = {
    "q5": {"label_col": "A_self", "trait": "Agreeableness"},
}

GRID_SEARCH_PARAMS = {
    "lr": [1e-4, 5e-5, 1e-5],
    "weight_decay": [5e-2, 1e-2, 1e-3, 1e-4],
    "dropout_rate": [0.0, 0.1, 0.3, 0.5],
    "hidden_dims": [
        (256,), (512,), (256, 64), (128, 32),
        (512, 256, 64), (1024, 512, 256)
    ],
}


# =============================
# 2. 工具函数
# =============================
def setup_logging(q_dir, q_type, combo_str, seed):
    log_name = f"log_{q_type}_{combo_str}_seed_{seed}.log"
    log_path = os.path.join(q_dir, log_name)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger(f"{q_type}_{combo_str}_{seed}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_bins(y, n_bins=5):
    return pd.cut(y, bins=n_bins, labels=False, duplicates="drop")


# =============================
# 3. 数据集与模型定义
# =============================
class DynamicModalDataset(Dataset):
    def __init__(self, df, active_modalities, q_type, label_col=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.active_modalities = active_modalities
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

        raise FileNotFoundError(f"Missing {mod_key} for {sid}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row["id"])

        embs = [self._load_emb(m, sid) for m in self.active_modalities]
        feat = torch.from_numpy(np.concatenate(embs)).float()

        if self.is_test or self.label_col is None:
            label = torch.tensor(0.0).float()
        else:
            label = torch.tensor(row[self.label_col]).float()

        return sid, feat, label


class RegressionHead(nn.Module):
    def __init__(self, in_dim, h_dims, dr):
        super().__init__()
        layers = []
        for h in h_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dr)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


# =============================
# 4. 训练与评估
# =============================
def train_eval_flow(model, loaders, transformer, lr, wd):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    best_trans_mse = float("inf")
    best_orig_mse = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        for _, xb, yb in loaders["train"]:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        pred_list, label_list = [], []

        with torch.no_grad():
            for _, xb, yb in loaders["val"]:
                xb = xb.to(DEVICE)
                pred_list.append(model(xb).cpu().numpy())
                label_list.append(yb.numpy())

        pred_trans = np.concatenate(pred_list)
        label_trans = np.concatenate(label_list)

        trans_mse = mean_squared_error(label_trans, pred_trans)

        if trans_mse < best_trans_mse:
            pred_orig = transformer.inverse_transform(pred_trans.reshape(-1, 1)).reshape(-1)
            label_orig = transformer.inverse_transform(label_trans.reshape(-1, 1)).reshape(-1)

            best_trans_mse = trans_mse
            best_orig_mse = mean_squared_error(label_orig, pred_orig)
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            break

    return best_trans_mse, best_orig_mse, best_state


# =============================
# 5. 主程序
# =============================
def main():
    _ = pd.read_csv(TEMPLATE_PATH)
    config = QUESTION_CONFIGS[CHOSEN_QTYPE]
    label_col = config["label_col"]

    train_data = pd.read_csv(TRAIN_CSV_PATH)
    val_data = pd.read_csv(VAL_CSV_PATH)
    all_data = pd.concat([train_data, val_data], ignore_index=True)

    all_data["bins"] = make_bins(all_data[label_col], n_bins=5)

    keys = list(MODALITY_MAP.keys())
    combos = [list(c) for r in range(1, len(keys) + 1) for c in combinations(keys, r)]

    for seed in SEEDS:
        set_seed(seed)

        q_dir_name = f"{CHOSEN_QTYPE}_{config['trait']}_seed_{seed}"
        q_dir = os.path.join(ROOT_OUTPUT_DIR, q_dir_name)
        os.makedirs(q_dir, exist_ok=True)

        seed_model_save_dir = os.path.join(MODEL_SAVE_DIR, f"seed_{seed}")
        os.makedirs(seed_model_save_dir, exist_ok=True)

        for combo in combos:
            combo_str = "_".join(combo)
            logger = setup_logging(q_dir, CHOSEN_QTYPE, combo_str, seed)
            logger.info(f"Dimension: {CHOSEN_QTYPE} | Seed: {seed} | Modality: {combo_str} | Start")

            input_dim = 1536 * len(combo)
            param_grid = list(product(
                GRID_SEARCH_PARAMS["lr"],
                GRID_SEARCH_PARAMS["weight_decay"],
                GRID_SEARCH_PARAMS["dropout_rate"],
                GRID_SEARCH_PARAMS["hidden_dims"]
            ))

            best_mse = float("inf")
            best_hparams = None

            # =============================
            # 网格搜索：5-fold CV + Yeo-Johnson
            # =============================
            for lr, wd, dr, hds in tqdm(param_grid, desc=f"Grid {combo_str} Seed {seed}"):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                cv_mses = []

                for fold, (t_idx, v_idx) in enumerate(skf.split(all_data, all_data["bins"])):
                    fold_train = all_data.iloc[t_idx].copy()
                    fold_val = all_data.iloc[v_idx].copy()

                    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                    fold_train["y_trans"] = transformer.fit_transform(fold_train[[label_col]]).reshape(-1)
                    fold_val["y_trans"] = transformer.transform(fold_val[[label_col]]).reshape(-1)

                    loaders = {
                        "train": DataLoader(
                            DynamicModalDataset(fold_train, combo, CHOSEN_QTYPE, "y_trans"),
                            batch_size=BATCH_SIZE,
                            shuffle=True
                        ),
                        "val": DataLoader(
                            DynamicModalDataset(fold_val, combo, CHOSEN_QTYPE, "y_trans"),
                            batch_size=BATCH_SIZE,
                            shuffle=False
                        )
                    }

                    model = RegressionHead(input_dim, hds, dr).to(DEVICE)
                    _, orig_mse, _ = train_eval_flow(model, loaders, transformer, lr, wd)
                    cv_mses.append(orig_mse)

                avg_mse = float(np.mean(cv_mses))

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_hparams = {
                        "lr": lr,
                        "wd": wd,
                        "dr": dr,
                        "hds": hds,
                        "batch_size": BATCH_SIZE,
                    }

            logger.info(f"Best Average CV MSE: {best_mse:.6f} | Params: {best_hparams}")

            # =============================
            # 使用最优超参数重新训练 5 folds，保存全部 fold 用于 ensemble
            # =============================
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_mses = []

            for fold, (t_idx, v_idx) in enumerate(skf.split(all_data, all_data["bins"])):
                fold_train = all_data.iloc[t_idx].copy()
                fold_val = all_data.iloc[v_idx].copy()

                transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                fold_train["y_trans"] = transformer.fit_transform(fold_train[[label_col]]).reshape(-1)
                fold_val["y_trans"] = transformer.transform(fold_val[[label_col]]).reshape(-1)

                loaders = {
                    "train": DataLoader(
                        DynamicModalDataset(fold_train, combo, CHOSEN_QTYPE, "y_trans"),
                        batch_size=BATCH_SIZE,
                        shuffle=True
                    ),
                    "val": DataLoader(
                        DynamicModalDataset(fold_val, combo, CHOSEN_QTYPE, "y_trans"),
                        batch_size=BATCH_SIZE,
                        shuffle=False
                    )
                }

                model = RegressionHead(input_dim, best_hparams["hds"], best_hparams["dr"]).to(DEVICE)
                _, orig_mse, best_state = train_eval_flow(
                    model,
                    loaders,
                    transformer,
                    best_hparams["lr"],
                    best_hparams["wd"]
                )

                fold_mses.append(orig_mse)

                model_save_path = os.path.join(
                    seed_model_save_dir,
                    f"model_{combo_str}_fold_{fold}.pth"
                )

                torch.save({
                    "state_dict": best_state,
                    "transformer": transformer,
                    "config": config,
                    "hparams": best_hparams,
                    "combo": combo,
                    "fold": fold,
                    "fold_mse": orig_mse,
                    "label_transform": "yeo-johnson",
                }, model_save_path)

                logger.info(f"Saved Fold {fold} | MSE: {orig_mse:.6f} | Path: {model_save_path}")

            logger.info(f"5-Fold Ensemble CV MSE: {np.mean(fold_mses):.6f}")
            logger.info(f"Fold MSEs: {fold_mses}")


if __name__ == "__main__":
    main()
