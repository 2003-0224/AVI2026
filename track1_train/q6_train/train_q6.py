# with meta-information
import os, sys, random, copy, logging, torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

# 模态路径映射
MODALITY_MAP = {
    "t": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_t"),
    "a": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_a"),
    "v": os.path.join(BASE_DIR, f"gemini_embedding/gemini_embeddings_{CHOSEN_QTYPE}_v"),
}

MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, CHOSEN_QTYPE)
TEMPLATE_PATH = os.path.join(BASE_DIR, "submission.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 实验与网格搜索参数
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10
SEEDS = [42, 1023, 2026]

QUESTION_CONFIGS = {
    "q6": {"label_col": "C_self", "trait": "Conscientiousness"},
}

GRID_SEARCH_PARAMS = {
    "lr": [1e-4, 5e-5, 1e-5],
    "weight_decay": [0, 5e-2, 1e-2, 1e-3],
    "dropout_rate": [0.0, 0.1, 0.3, 0.5],
    "hidden_dims": [
        (128,), (256,), (512,),
        (256, 64), (128, 32), (512, 256, 64),
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
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger(f"{q_type}_{combo_str}_{seed}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================
# 3. 数据集与模型定义
# =============================
class DynamicModalDataset(Dataset):
    def __init__(self, df, active_modalities, q_type, label_col=None, is_test=False):
        self.df, self.active_modalities, self.q_type = df, active_modalities, q_type
        self.label_col, self.is_test = label_col, is_test

    def _load_emb(self, mod_key, sid):
        base_dir = MODALITY_MAP[mod_key]
        fname = f"{sid}_{self.q_type}.npz"
        for sub in ["train", "val", "test", ""]:
            path = os.path.join(base_dir, sub, fname)
            if os.path.exists(path):
                with np.load(path) as d: return d["embedding"]
        raise FileNotFoundError(f"Missing {mod_key} for {sid}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row['id'])
        embs = [self._load_emb(m, sid) for m in self.active_modalities]
        feat = torch.from_numpy(np.concatenate(embs)).float()
        label = torch.tensor(row[self.label_col] if not self.is_test and self.label_col else 0.0).float()
        return sid, feat, label


class RegressionHead(nn.Module):
    def __init__(self, in_dim, h_dims, dr):
        super().__init__()
        layers = []
        for h in h_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dr)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).view(-1)


# =============================
# 4. 训练与评估流
# =============================
def train_eval_flow(model, loaders, scaler, lr, wd):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best_norm_mse, best_orig_mse, p_cnt = float("inf"), float("inf"), 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for _, xb, yb in loaders['train']:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        p_list, l_list = [], []
        with torch.no_grad():
            for _, xb, yb in loaders['val']:
                p_list.append(model(xb.to(DEVICE)).cpu().numpy())
                l_list.append(yb.numpy())

        pn, ln = np.concatenate(p_list), np.concatenate(l_list)
        norm_mse = mean_squared_error(ln, pn)

        if norm_mse < best_norm_mse:
            po = scaler.inverse_transform(pn.reshape(-1, 1))
            lo = scaler.inverse_transform(ln.reshape(-1, 1))
            best_norm_mse, best_orig_mse, p_cnt = norm_mse, mean_squared_error(lo, po), 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            p_cnt += 1
        if p_cnt >= PATIENCE: break
    return best_norm_mse, best_orig_mse, best_state


# =============================
# 5. 主程序
# =============================
def main():
    template_df = pd.read_csv(TEMPLATE_PATH)
    config = QUESTION_CONFIGS[CHOSEN_QTYPE]

    # 读取独立的 train 和 val，合并并重新构建索引避免重复
    train_data = pd.read_csv(TRAIN_CSV_PATH)
    val_data = pd.read_csv(VAL_CSV_PATH)
    all_data = pd.concat([train_data, val_data], ignore_index=True)

    # 统一标签标准化
    scaler = StandardScaler()
    all_data['y_norm'] = scaler.fit_transform(all_data[[config['label_col']]])

    # 将连续值标签切分为5个等间距桶，提供给 StratifiedKFold 做分层划分依据
    all_data['bins'] = pd.cut(all_data[config['label_col']], bins=5, labels=False)

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
            param_grid = list(product(GRID_SEARCH_PARAMS["lr"], GRID_SEARCH_PARAMS["weight_decay"],
                                      GRID_SEARCH_PARAMS["dropout_rate"], GRID_SEARCH_PARAMS["hidden_dims"]))

            best_mse, best_hparams = float("inf"), None

            # --- 网格搜索阶段：使用分桶后的合并数据进行 5 折交叉验证 ---
            for lr, wd, dr, hds in tqdm(param_grid, desc=f"Grid {combo_str} Seed {seed}"):
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                cv_mses = []
                for t_idx, v_idx in skf.split(all_data, all_data['bins']):
                    loaders = {
                        'train': DataLoader(DynamicModalDataset(all_data.iloc[t_idx], combo, CHOSEN_QTYPE, 'y_norm'),
                                            batch_size=BATCH_SIZE, shuffle=True),
                        'val': DataLoader(DynamicModalDataset(all_data.iloc[v_idx], combo, CHOSEN_QTYPE, 'y_norm'),
                                          batch_size=BATCH_SIZE)
                    }
                    model = RegressionHead(input_dim, hds, dr).to(DEVICE)
                    _, f_orig_mse, _ = train_eval_flow(model, loaders, scaler, lr, wd)
                    cv_mses.append(f_orig_mse)

                avg_mse = np.mean(cv_mses)
                if avg_mse < best_mse:
                    best_mse, best_hparams = avg_mse, {"lr": lr, "wd": wd, "dr": dr, "hds": hds}

            logger.info(f"Best Average CV MSE: {best_mse:.6f} | Params: {best_hparams}")

            # --- 5折选最优折逻辑 ---
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            test_loader = DataLoader(DynamicModalDataset(template_df, combo, CHOSEN_QTYPE, is_test=True),
                                     batch_size=BATCH_SIZE)

            best_fold_mse = float("inf")
            best_fold_state = None
            best_fold_idx = -1

            for fold, (t_idx, v_idx) in enumerate(skf.split(all_data, all_data['bins'])):
                loaders = {
                    'train': DataLoader(DynamicModalDataset(all_data.iloc[t_idx], combo, CHOSEN_QTYPE, 'y_norm'),
                                        batch_size=BATCH_SIZE, shuffle=True),
                    'val': DataLoader(DynamicModalDataset(all_data.iloc[v_idx], combo, CHOSEN_QTYPE, 'y_norm'),
                                      batch_size=BATCH_SIZE)
                }
                model = RegressionHead(input_dim, best_hparams["hds"], best_hparams["dr"]).to(DEVICE)
                _, f_orig_mse, b_state = train_eval_flow(model, loaders, scaler, best_hparams["lr"], best_hparams["wd"])

                if f_orig_mse < best_fold_mse:
                    best_fold_mse = f_orig_mse
                    best_fold_state = copy.deepcopy(b_state)
                    best_fold_idx = fold

            logger.info(f"Best Fold: Fold {best_fold_idx} | Val MSE: {best_fold_mse:.6f}")
            best_hparams["batch_size"] = BATCH_SIZE

            # 保存单最优折权重
            model_save_path = os.path.join(seed_model_save_dir, f"model_{combo_str}.pth")
            torch.save({
                'state_dict': best_fold_state,
                'scaler': scaler,
                'config': config,
                'hparams': best_hparams,
            }, model_save_path)

            print("Model saved in path: %s" % model_save_path)


if __name__ == "__main__":
    main()