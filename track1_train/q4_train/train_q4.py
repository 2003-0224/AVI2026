import os
import random
import copy
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from itertools import product, combinations

from model import AttentionFusionRegressor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_hparams_to_json(hparams, output_path):
    try:
        with open(output_path, "w") as f:
            json.dump(hparams, f, indent=4)
    except TypeError:
        hparams_serializable = {k: str(v) for k, v in hparams.items()}
        with open(output_path, "w") as f:
            json.dump(hparams_serializable, f, indent=4)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class AttentionMultiModalDataset(Dataset):
    def __init__(self, df, emb_dirs, question_type, active_modalities, label_col=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.emb_dirs = emb_dirs
        self.active_modalities = active_modalities
        self.question_type = question_type
        self.label_col = label_col
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def _load_embedding(self, modality, sample_id):
        base_dir = self.emb_dirs[modality]
        filename = f"{sample_id}_{self.question_type}.npz"
        for sub_dir in ["train", "val", "test", ""]:
            potential_path = os.path.join(base_dir, sub_dir, filename) if sub_dir else os.path.join(base_dir, filename)
            if os.path.exists(potential_path):
                with np.load(potential_path) as data:
                    return data["embedding"].astype(np.float32)
        raise FileNotFoundError(f"Embedding file '{filename}' not found for modality '{modality}'.")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        embeddings = [self._load_embedding(m, sample_id) for m in self.active_modalities]
        stacked_embedding = np.stack(embeddings, axis=0)

        label = row[self.label_col] if not self.is_test and self.label_col is not None else 0.0
        return (
            str(sample_id),
            torch.tensor(stacked_embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def train_one_epoch(model, loader, criterion, optimizer, ema=None):
    model.train()
    for _, xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb).view(-1)
        loss = criterion(preds, yb.view(-1))
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)


def evaluate(model, loader, criterion, transformer, ema=None):
    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    all_preds_trans, all_labels_trans = [], []

    with torch.no_grad():
        for _, xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).view(-1)
            all_preds_trans.append(preds.cpu().numpy())
            all_labels_trans.append(yb.numpy())

    if ema is not None:
        ema.restore(model)

    all_preds_trans = np.concatenate(all_preds_trans)
    all_labels_trans = np.concatenate(all_labels_trans)

    trans_mse = mean_squared_error(all_labels_trans, all_preds_trans)

    all_preds_orig = transformer.inverse_transform(all_preds_trans.reshape(-1, 1)).flatten()
    all_labels_orig = transformer.inverse_transform(all_labels_trans.reshape(-1, 1)).flatten()
    orig_mse = mean_squared_error(all_labels_orig, all_preds_orig)

    return trans_mse, orig_mse


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_EMB_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "gemini_embeding"))
BASE_EMB_DIRS = {
    "text": os.path.join(BASE_EMB_ROOT, "gemini_embeddings_q4_t"),
    "audio": os.path.join(BASE_EMB_ROOT, "gemini_embeddings_q4_a"),
    "video": os.path.join(BASE_EMB_ROOT, "gemini_embeddings_q4_v"),
}

BASE_LABEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BASE_MODEL_SAVE_DIR = "saved_models"
OUTPUT_DIR = "all_combinations_results_q4/"
TEMPLATE_PATH = os.path.join(BASE_LABEL_DIR, "template.csv")

INPUT_DIM = 1536
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMA_DECAY = 0.999

QUESTION_CONFIGS = {
    "q4": {"label_col": "E_self", "trait": "Extraversion"},
}

GRID_SEARCH_PARAMS = {
    "lr": [1e-3, 1e-4, 5e-5],
    "weight_decay": [1e-2, 1e-3, 1e-4, 1e-5],
    "dropout_rate": [0.1, 0.3, 0.5],
    "hidden_dims": [
        (256, 64), (512, 128), (128.32), (512, 128, 32)
    ],
    "num_heads": [2, 4, 8],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    set_seed()
    template_df = pd.read_csv(TEMPLATE_PATH)

    all_mods = list(BASE_EMB_DIRS.keys())
    mod_combinations = []
    for r in range(1, len(all_mods) + 1):
        mod_combinations.extend([list(c) for c in combinations(all_mods, r)])

    summary_results = {}

    for q_type, config in QUESTION_CONFIGS.items():
        print(f"\n{'=' * 40}")
        print(f" 正在处理维度: {config['trait']} ({q_type}) ")
        print(f"{'=' * 40}")

        train_csv = os.path.join(BASE_LABEL_DIR, "train_expanded.csv")
        val_csv = os.path.join(BASE_LABEL_DIR, "val_expanded.csv")
        all_data = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], ignore_index=True)
        all_data = all_data[all_data["question_type"] == q_type].copy()

        label_col = config["label_col"]
        trans_label_col = label_col + "_yj"

        all_data["bins"] = pd.cut(all_data[label_col], bins=5, labels=False, duplicates="drop")

        for active_mods in mod_combinations:
            combo_name = "_".join(active_mods)
            print(f"\n>>> 正在探讨模态组合: {combo_name} <<<")

            print("\n--- Pass 1: 网格搜索寻找最佳超参数 ---")
            param_grid = list(product(
                GRID_SEARCH_PARAMS["lr"],
                GRID_SEARCH_PARAMS["weight_decay"],
                GRID_SEARCH_PARAMS["dropout_rate"],
                GRID_SEARCH_PARAMS["hidden_dims"],
                GRID_SEARCH_PARAMS["num_heads"],
            ))

            best_overall_orig_mse = float("inf")
            best_hparams = None

            for lr, wd, dr, hds, n_heads in tqdm(param_grid, desc=f"Grid Search [{combo_name}]"):
                cv_orig_mses = []
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                for train_idx, val_idx in skf.split(all_data, all_data["bins"]):
                    train_sub = all_data.iloc[train_idx].copy()
                    val_sub = all_data.iloc[val_idx].copy()

                    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                    train_sub[trans_label_col] = transformer.fit_transform(train_sub[[label_col]]).reshape(-1)
                    val_sub[trans_label_col] = transformer.transform(val_sub[[label_col]]).reshape(-1)

                    train_ds = AttentionMultiModalDataset(
                        train_sub, BASE_EMB_DIRS, q_type, active_mods, label_col=trans_label_col
                    )
                    val_ds = AttentionMultiModalDataset(
                        val_sub, BASE_EMB_DIRS, q_type, active_mods, label_col=trans_label_col
                    )

                    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

                    model = AttentionFusionRegressor(INPUT_DIM, n_heads, hds, dr).to(DEVICE)
                    ema = EMA(model, decay=EMA_DECAY)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                    criterion = nn.MSELoss()

                    best_val_trans_mse = float("inf")
                    best_fold_orig_mse = float("inf")
                    patience_cnt = 0

                    for epoch in range(1, EPOCHS + 1):
                        train_one_epoch(model, train_loader, criterion, optimizer, ema)
                        val_trans_mse, val_orig_mse = evaluate(model, val_loader, criterion, transformer, ema)

                        if val_trans_mse < best_val_trans_mse:
                            best_val_trans_mse = val_trans_mse
                            best_fold_orig_mse = val_orig_mse
                            patience_cnt = 0
                        else:
                            patience_cnt += 1

                        if patience_cnt >= PATIENCE:
                            break

                    cv_orig_mses.append(best_fold_orig_mse)

                avg_cv_orig_mse = float(np.mean(cv_orig_mses))

                if avg_cv_orig_mse < best_overall_orig_mse:
                    best_overall_orig_mse = avg_cv_orig_mse
                    best_hparams = {
                        "lr": lr,
                        "wd": wd,
                        "dr": dr,
                        "hds": hds,
                        "n_heads": n_heads,
                    }

            print(f"[{combo_name}] 最佳平均 MSE: {best_overall_orig_mse:.6f} | 参数: {best_hparams}")
            summary_results[combo_name] = best_overall_orig_mse

            hparams_output_path = os.path.join(OUTPUT_DIR, f"best_hparams_{q_type}_{combo_name}.json")
            save_hparams_to_json(best_hparams, hparams_output_path)

            print("\n--- Pass 2: 使用最优超参数训练五折模型并保存 ---")
            fold_models = []
            fold_orig_mses = []

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, all_data["bins"])):
                train_sub = all_data.iloc[train_idx].copy()
                val_sub = all_data.iloc[val_idx].copy()

                transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                train_sub[trans_label_col] = transformer.fit_transform(train_sub[[label_col]]).reshape(-1)
                val_sub[trans_label_col] = transformer.transform(val_sub[[label_col]]).reshape(-1)

                train_ds = AttentionMultiModalDataset(
                    train_sub, BASE_EMB_DIRS, q_type, active_mods, label_col=trans_label_col
                )
                val_ds = AttentionMultiModalDataset(
                    val_sub, BASE_EMB_DIRS, q_type, active_mods, label_col=trans_label_col
                )

                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

                model = AttentionFusionRegressor(
                    INPUT_DIM,
                    best_hparams["n_heads"],
                    best_hparams["hds"],
                    best_hparams["dr"],
                ).to(DEVICE)

                ema = EMA(model, decay=EMA_DECAY)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=best_hparams["lr"],
                    weight_decay=best_hparams["wd"],
                )
                criterion = nn.MSELoss()

                best_val_trans_mse = float("inf")
                best_fold_orig_mse = float("inf")
                best_model_state_in_fold = None
                patience_cnt = 0

                for epoch in range(1, EPOCHS + 1):
                    train_one_epoch(model, train_loader, criterion, optimizer, ema)
                    val_trans_mse, val_orig_mse = evaluate(model, val_loader, criterion, transformer, ema)

                    if val_trans_mse < best_val_trans_mse:
                        best_val_trans_mse = val_trans_mse
                        best_fold_orig_mse = val_orig_mse

                        ema.apply_shadow(model)
                        best_model_state_in_fold = copy.deepcopy(model.state_dict())
                        ema.restore(model)

                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                    if patience_cnt >= PATIENCE:
                        break

                fold_orig_mses.append(best_fold_orig_mse)

                model_save_path = os.path.join(OUTPUT_DIR, f"model_{q_type}_{combo_name}_fold_{fold}.pth")
                best_hparams["batch_size"] = BATCH_SIZE

                torch.save({
                    "state_dict": best_model_state_in_fold,
                    "transformer": transformer,
                    "config": config,
                    "hparams": best_hparams,
                    "active_mods": active_mods,
                    "fold": fold,
                    "fold_mse": best_fold_orig_mse,
                    "label_transform": "yeo-johnson",
                }, model_save_path)

                fold_models.append((copy.deepcopy(best_model_state_in_fold), transformer))

                print(f"Fold {fold} saved | MSE={best_fold_orig_mse:.6f} | Path={model_save_path}")

            print(f"[{combo_name}] 五折平均 MSE: {np.mean(fold_orig_mses):.6f}")
            print(f"[{combo_name}] 各折 MSE: {fold_orig_mses}")

            print("\n--- Pass 3: 五折 Ensemble 预测 ---")
            test_ds = AttentionMultiModalDataset(template_df, BASE_EMB_DIRS, q_type, active_mods, is_test=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            fold_preds_orig = []

            for fold, (state_dict, transformer) in enumerate(fold_models):
                model = AttentionFusionRegressor(
                    INPUT_DIM,
                    best_hparams["n_heads"],
                    best_hparams["hds"],
                    best_hparams["dr"],
                ).to(DEVICE)

                model.load_state_dict(state_dict)
                model.eval()

                preds_trans = []
                with torch.no_grad():
                    for _, xb, _ in test_loader:
                        xb = xb.to(DEVICE)
                        outputs = model(xb).view(-1)
                        preds_trans.append(outputs.cpu().numpy())

                preds_trans = np.concatenate(preds_trans)
                preds_orig = transformer.inverse_transform(preds_trans.reshape(-1, 1)).flatten()
                fold_preds_orig.append(preds_orig)

            avg_preds_orig = np.mean(np.stack(fold_preds_orig, axis=0), axis=0)

            temp_output_df = template_df.copy()
            preds_dict = {str(row.id): pred for row, pred in zip(temp_output_df.itertuples(), avg_preds_orig)}
            temp_output_df[config["trait"]] = temp_output_df["id"].astype(str).map(preds_dict)

            final_output_path = os.path.join(OUTPUT_DIR, f"{best_overall_orig_mse:.6f}_{q_type}_{combo_name}_ensemble.csv")
            temp_output_df.to_csv(final_output_path, index=False)
            print(f"[{combo_name}] 五折 Ensemble 预测结果已保存至: {final_output_path}")

    print("\n" + "=" * 50)
    print(" 各模态组合效果汇总 (按 MSE 从小到大排序)")
    print("=" * 50)
    sorted_summary = sorted(summary_results.items(), key=lambda x: x[1])
    for rank, (combo, mse) in enumerate(sorted_summary, 1):
        print(f"Top {rank}: {combo.ljust(20)} | MSE = {mse:.6f}")


if __name__ == "__main__":
    main()
