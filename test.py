import os
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 定义问题类型与目标列名的映射关系
TRAIT_MAP = {
    "q3": "Honesty-Humility",
    "q4": "Extraversion",
    "q5": "Agreeableness",
    "q6": "Conscientiousness"
}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class RegressionHead(nn.Module):
    def __init__(self, in_dim, h_dims, dr, use_ln=False):
        super().__init__()
        layers = []
        for h in h_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_ln:
                layers.append(nn.LayerNorm(h))
            layers.extend([nn.ReLU(), nn.Dropout(dr)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


class TestDataset(Dataset):
    def __init__(self, df, active_modalities, q_type, modality_paths):
        self.df = df
        self.active_modalities = active_modalities
        self.q_type = q_type
        self.modality_paths = modality_paths

    def _load_emb(self, mod_key, sid):
        base_dir = self.modality_paths[mod_key]
        fname = f"{sid}_{self.q_type}.npz"
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            with np.load(path) as d:
                return d["embedding"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row['id'])
        embs = [self._load_emb(m, sid) for m in self.active_modalities]
        feat = torch.from_numpy(np.concatenate(embs)).float()
        return sid, feat


def run_single_task_inference(model_path, test_df, q_type, modality_paths):
    print(f"\n=====> 正在加载 [{q_type}] 模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    # 1. 加载 Checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict']
    scaler = checkpoint['scaler']
    hparams = checkpoint.get('hparams')

    # 2. 自动检测模型是否包含 LayerNorm
    use_ln = any("LayerNorm" in k or ".1.bias" in k for k in state_dict.keys())
    print(f"模型结构自适应: {'启用' if use_ln else '禁用'} LayerNorm")

    # 3. 初始化模型
    first_key = list(state_dict.keys())[0]
    input_dim = state_dict[first_key].shape[1]
    model = RegressionHead(input_dim, hparams['hds'], hparams['dr'], use_ln=use_ln).to(DEVICE)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = {k.replace('net.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    model.eval()

    # 4. 从模型文件名解析当前模型使用的模态顺序
    file_name = os.path.basename(model_path)
    name_parts = file_name.replace(".pth", "").split("_")
    active_mods = [m for m in name_parts if m in ["t", "a", "v"]]
    if not active_mods:
        # 如果从文件名无法正确解析，则使用维度推断（默认按 t, a, v 顺序）
        num_mods = input_dim // 1536
        active_mods = ["t", "a", "v"][:num_mods]

    print(f"推理模态: {active_mods} | 维度类型: {q_type} | 输入总维度: {input_dim}")

    # 5. 准备数据加载器
    dataset = TestDataset(test_df, active_mods, q_type, modality_paths)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 6. 执行推理
    ids, preds = [], []
    with torch.no_grad():
        for s_ids, xb in loader:
            out = model(xb.to(DEVICE)).cpu().numpy()
            out_orig = scaler.inverse_transform(out.reshape(-1, 1)).flatten()
            ids.extend(s_ids)
            preds.extend(out_orig)

    res_df = pd.DataFrame({"id": ids, TRAIT_MAP[q_type]: preds})
    # 确保 id 列类型一致以防后续 merge 错位
    res_df["id"] = res_df["id"].astype(str)
    return res_df


def main():
    parser = argparse.ArgumentParser(description="多任务串行推理脚本")
    # 基础输入输出路径
    parser.add_argument("--test_data_path", type=str, required=True, help="测试集 submission.csv 模板路径")
    parser.add_argument("--output_result_path", type=str, required=True, help="保存预测结果的路径")

    # 4个维度的模型权重路径
    parser.add_argument("--model_q3", type=str, required=True, help="q3 (Honesty-Humility) 模型路径")
    parser.add_argument("--model_q4", type=str, required=True, help="q4 (Extraversion) 模型路径")
    parser.add_argument("--model_q5", type=str, required=True, help="q5 (Agreeableness) 模型路径")
    parser.add_argument("--model_q6", type=str, required=True, help="q6 (Conscientiousness) 模型路径")

    # 各个问题对应的tav模态目录路径
    parser.add_argument("--q3_t_dir", type=str, required=True)
    parser.add_argument("--q3_a_dir", type=str, required=True)
    parser.add_argument("--q3_v_dir", type=str, required=True)

    parser.add_argument("--q4_t_dir", type=str, required=True)
    parser.add_argument("--q4_a_dir", type=str, required=True)
    parser.add_argument("--q4_v_dir", type=str, required=True)

    parser.add_argument("--q5_t_dir", type=str, required=True)
    parser.add_argument("--q5_a_dir", type=str, required=True)
    parser.add_argument("--q5_v_dir", type=str, required=True)

    parser.add_argument("--q6_t_dir", type=str, required=True)
    parser.add_argument("--q6_a_dir", type=str, required=True)
    parser.add_argument("--q6_v_dir", type=str, required=True)

    args = parser.parse_args()

    # 读取测试集基础表格
    base_test_df = pd.read_csv(args.test_data_path)
    base_test_df["id"] = base_test_df["id"].astype(str)

    # 最终用于保存合并结果的 DataFrame，先保留完整的 id 序列
    submission_df = base_test_df[["id"]].copy()

    # 封装各维度的特征路径映射
    modality_configs = {
        "q3": {"t": args.q3_t_dir, "a": args.q3_a_dir, "v": args.q3_v_dir},
        "q4": {"t": args.q4_t_dir, "a": args.q4_a_dir, "v": args.q4_v_dir},
        "q5": {"t": args.q5_t_dir, "a": args.q5_a_dir, "v": args.q5_v_dir},
        "q6": {"t": args.q6_t_dir, "a": args.q6_a_dir, "v": args.q6_v_dir},
    }

    # 按照 q3, q4, q5, q6 的顺序执行测试
    task_sequence = [
        ("q3", args.model_q3),
        ("q4", args.model_q4),
        ("q5", args.model_q5),
        ("q6", args.model_q6)
    ]

    for q_type, model_path in task_sequence:
        try:
            res_df = run_single_task_inference(
                model_path=model_path,
                test_df=base_test_df,
                q_type=q_type,
                modality_paths=modality_configs[q_type]
            )
            # 通过 id 将当前维度预测结果合并进总表
            submission_df = pd.merge(submission_df, res_df, on="id", how="left")
            print(f"[{q_type}] 推理并合并成功。")
        except Exception as e:
            print(f"[{q_type}] 推理发生错误: {e}")
            raise e

    # 确保输出文件目录存在
    out_dir = os.path.dirname(args.output_result_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 写入最终的 submission.csv 中
    submission_df.to_csv(args.output_result_path, index=False)
    print(f"\n=====> 全维度测试完成！结果已成功保存至: {args.output_result_path}")


if __name__ == "__main__":
    main()
