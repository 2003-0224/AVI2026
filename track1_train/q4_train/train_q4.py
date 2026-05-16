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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from itertools import product, combinations
from model import RegressionHead, AttentionFusionRegressor


# =============================
# 2. 工具函数与 EMA 类
# =============================
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
        with open(output_path, 'w') as f:
            json.dump(hparams, f, indent=4)
    except TypeError as e:
        hparams_serializable = {k: str(v) for k, v in hparams.items()}
        with open(output_path, 'w') as f:
            json.dump(hparams_serializable, f, indent=4)


# 新增 EMA (指数移动平均) 类
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


# =============================
# 3. 数据集
# =============================
class AttentionMultiModalDataset(Dataset):
    def __init__(self, df, emb_dirs, question_type, active_modalities, label_col=None, is_test=False):
        self.df = df
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
                    return data["embedding"]
        raise FileNotFoundError(f"Embedding file '{filename}' not found for modality '{modality}'.")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['id']
        # 仅加载被激活的模态
        embeddings = [self._load_embedding(m, sample_id) for m in self.active_modalities]
        stacked_embedding = np.stack(embeddings, axis=0)
        label = row[self.label_col] if not self.is_test and self.label_col is not None else 0.0
        return str(sample_id), torch.tensor(stacked_embedding, dtype=torch.float32), torch.tensor(label,
                                                                                                  dtype=torch.float32)


# =============================
# 4. 核心训练/评估函数
# =============================
def train_one_epoch(model, loader, criterion, optimizer, ema=None):
    model.train()
    for _, xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb).view(-1)
        yb = yb.view(-1)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        # 每步更新 EMA 权重
        if ema is not None:
            ema.update(model)


def evaluate(model, loader, criterion, scaler, ema=None):
    # 评估时应用 EMA 权重
    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    all_preds_norm, all_labels_norm = [], []
    with torch.no_grad():
        for _, xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).view(-1)
            all_preds_norm.append(preds.cpu().numpy())
            all_labels_norm.append(yb.numpy())

    # 评估完成后恢复正常权重，以免影响后续训练
    if ema is not None:
        ema.restore(model)

    all_preds_norm = np.concatenate(all_preds_norm)
    all_labels_norm = np.concatenate(all_labels_norm)
    norm_mse = mean_squared_error(all_labels_norm, all_preds_norm)

    all_preds_orig = scaler.inverse_transform(all_preds_norm.reshape(-1, 1)).flatten()
    all_labels_orig = scaler.inverse_transform(all_labels_norm.reshape(-1, 1)).flatten()
    orig_mse = mean_squared_error(all_labels_orig, all_preds_orig)
    return norm_mse, orig_mse


# =============================
# 1. 全局配置
# =============================
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
EMA_DECAY = 0.999  # 新增 EMA 衰减率配置

QUESTION_CONFIGS = {
    "q4": {"label_col": "E_self", "trait": "Extraversion"},
}

GRID_SEARCH_PARAMS = {
    "lr": [1e-3, 1e-4, 5e-5],
    "weight_decay": [1e-2, 1e-3, 1e-4, 1e-5],
    "dropout_rate": [0.1, 0.3, 0.5],
    "hidden_dims": [
        (256, 64), (512, 128), (4096, 2048, 1024), (2048, 1024, 512), (1024, 512, 256),
    ],
    "num_heads": [4, 8, 16],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# 5. 主程序
# =============================
def main():
    set_seed()
    template_df = pd.read_csv(TEMPLATE_PATH)

    # 自动生成所有模态组合 (1个，2个，3个)
    all_mods = list(BASE_EMB_DIRS.keys())
    mod_combinations = []
    for r in range(1, len(all_mods) + 1):
        mod_combinations.extend([list(c) for c in combinations(all_mods, r)])

    summary_results = {}  # 用于最终对比不同模态组合的效果

    for q_type, config in QUESTION_CONFIGS.items():
        print(f"\n{'=' * 40}")
        print(f" 正在处理维度: {config['trait']} ({q_type}) ")
        print(f"{'=' * 40}")

        train_csv = os.path.join(BASE_LABEL_DIR, "train_expanded.csv")
        val_csv = os.path.join(BASE_LABEL_DIR, "val_expanded.csv")
        all_data = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], ignore_index=True)
        all_data = all_data[all_data["question_type"] == q_type].copy()

        label_col, norm_label_col = config['label_col'], config['label_col'] + '_norm'
        scaler = StandardScaler()
        all_data[norm_label_col] = scaler.fit_transform(all_data[[label_col]])
        all_data['bins'] = pd.cut(all_data[label_col], bins=5, labels=False)

        # ----------------------------------------------------
        # 外层大循环：遍历所有模态组合
        # ----------------------------------------------------
        for active_mods in mod_combinations:
            combo_name = "_".join(active_mods)
            print(f"\n>>> 正在探讨模态组合: {combo_name} <<<")

            print("\n--- Pass 1: 网格搜索寻找最佳超参数 ---")
            param_grid = list(product(GRID_SEARCH_PARAMS["lr"], GRID_SEARCH_PARAMS["weight_decay"],
                                      GRID_SEARCH_PARAMS["dropout_rate"], GRID_SEARCH_PARAMS["hidden_dims"],
                                      GRID_SEARCH_PARAMS["num_heads"]))
            best_overall_orig_mse, best_hparams = float("inf"), None

            for lr, wd, dr, hds, n_heads in tqdm(param_grid, desc=f"Grid Search [{combo_name}]"):
                cv_orig_mses = []
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                for train_idx, val_idx in skf.split(all_data, all_data['bins']):
                    train_sub, val_sub = all_data.iloc[train_idx], all_data.iloc[val_idx]
                    train_ds = AttentionMultiModalDataset(train_sub, BASE_EMB_DIRS, q_type, active_mods,
                                                          label_col=norm_label_col)
                    val_ds = AttentionMultiModalDataset(val_sub, BASE_EMB_DIRS, q_type, active_mods,
                                                        label_col=norm_label_col)
                    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

                    # 初始化模型和 EMA
                    model = AttentionFusionRegressor(INPUT_DIM, n_heads, hds, dr).to(DEVICE)
                    ema = EMA(model, decay=EMA_DECAY)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                    criterion = nn.MSELoss()

                    best_val_norm_mse, best_fold_orig_mse, patience_cnt = float("inf"), float("inf"), 0
                    for epoch in range(1, EPOCHS + 1):
                        train_one_epoch(model, train_loader, criterion, optimizer, ema)
                        val_norm_mse, val_orig_mse = evaluate(model, val_loader, criterion, scaler, ema)

                        if val_norm_mse < best_val_norm_mse:
                            best_val_norm_mse, best_fold_orig_mse, patience_cnt = val_norm_mse, val_orig_mse, 0
                        else:
                            patience_cnt += 1
                        if patience_cnt >= PATIENCE: break
                    cv_orig_mses.append(best_fold_orig_mse)

                avg_cv_orig_mse = np.mean(cv_orig_mses)
                if avg_cv_orig_mse < best_overall_orig_mse:
                    best_overall_orig_mse = avg_cv_orig_mse
                    best_hparams = {"lr": lr, "wd": wd, "dr": dr, "hds": hds, "n_heads": n_heads}

            print(f"[{combo_name}] 最佳平均 MSE: {best_overall_orig_mse:.6f} | 参数: {best_hparams}")
            summary_results[combo_name] = best_overall_orig_mse
            hparams_output_path = os.path.join(OUTPUT_DIR, f"best_hparams_{q_type}_{combo_name}.json")
            save_hparams_to_json(best_hparams, hparams_output_path)

            print("\n--- Pass 2: 重新训练并选出最好的折保存权重 ---")
            best_model_overall_state = None
            best_val_norm_mse_across_folds = float("inf")
            best_fold_idx = -1

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, all_data['bins'])):
                train_sub, val_sub = all_data.iloc[train_idx], all_data.iloc[val_idx]
                train_ds = AttentionMultiModalDataset(train_sub, BASE_EMB_DIRS, q_type, active_mods,
                                                      label_col=norm_label_col)
                val_ds = AttentionMultiModalDataset(val_sub, BASE_EMB_DIRS, q_type, active_mods,
                                                    label_col=norm_label_col)
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

                model = AttentionFusionRegressor(INPUT_DIM, best_hparams["n_heads"], best_hparams["hds"],
                                                 best_hparams["dr"]).to(DEVICE)
                ema = EMA(model, decay=EMA_DECAY)
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_hparams["lr"],
                                              weight_decay=best_hparams["wd"])
                criterion = nn.MSELoss()

                best_val_norm_mse_in_fold = float("inf")
                best_model_state_in_fold = None

                for epoch in range(1, EPOCHS + 1):
                    train_one_epoch(model, train_loader, criterion, optimizer, ema)
                    val_norm_mse, _ = evaluate(model, val_loader, criterion, scaler, ema)

                    if val_norm_mse < best_val_norm_mse_in_fold:
                        best_val_norm_mse_in_fold = val_norm_mse

                        # 应用并缓存 EMA 权重
                        ema.apply_shadow(model)
                        best_model_state_in_fold = copy.deepcopy(model.state_dict())
                        ema.restore(model)

                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                    if patience_cnt >= PATIENCE: break

                if best_val_norm_mse_in_fold < best_val_norm_mse_across_folds:
                    best_val_norm_mse_across_folds = best_val_norm_mse_in_fold
                    best_model_overall_state = best_model_state_in_fold
                    best_fold_idx = fold

            print(f"选出最优折为: Fold {best_fold_idx}, 正在将其权重存盘...")
            model_save_path = os.path.join(OUTPUT_DIR, f"model_{q_type}_{combo_name}_best.pth")
            best_hparams["batch_size"] = BATCH_SIZE
            torch.save({
                'state_dict': best_model_overall_state,
                'scaler': scaler,
                'config': config,
                'hparams': best_hparams,
            }, model_save_path)

            print("\n--- Pass 3: 进行预测 ---")
            test_ds = AttentionMultiModalDataset(template_df, BASE_EMB_DIRS, q_type, active_mods, is_test=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

            # 直接加载那一个最好的折进行单模型推理
            model = AttentionFusionRegressor(INPUT_DIM, best_hparams["n_heads"], best_hparams["hds"],
                                             best_hparams["dr"]).to(DEVICE)
            model.load_state_dict(best_model_overall_state)
            model.eval()

            test_preds_norm = []
            with torch.no_grad():
                for _, xb, _ in test_loader:
                    xb = xb.to(DEVICE)
                    outputs = model(xb).view(-1)
                    test_preds_norm.append(outputs.cpu().numpy())

            avg_preds_norm = np.concatenate(test_preds_norm)
            avg_preds_orig = scaler.inverse_transform(avg_preds_norm.reshape(-1, 1)).flatten()

            # 生成对应的预测结果 CSV
            temp_output_df = template_df.copy()
            preds_dict = {str(row.id): pred for row, pred in zip(temp_output_df.itertuples(), avg_preds_orig)}
            temp_output_df[config['trait']] = temp_output_df['id'].astype(str).map(preds_dict)

            final_output_path = os.path.join(OUTPUT_DIR, f"{best_overall_orig_mse:.6f}_{q_type}_{combo_name}.csv")
            temp_output_df.to_csv(final_output_path, index=False)
            print(f"[{combo_name}] 预测结果已保存至: {final_output_path}")

    # =============================
    # 输出所有模态组合的效果对比报告
    # =============================
    print("\n" + "=" * 50)
    print(" 各模态组合效果汇总 (按 MSE 从小到大排序)")
    print("=" * 50)
    sorted_summary = sorted(summary_results.items(), key=lambda x: x[1])
    for rank, (combo, mse) in enumerate(sorted_summary, 1):
        print(f"Top {rank}: {combo.ljust(20)} | MSE = {mse:.6f}")


if __name__ == "__main__":
    main()