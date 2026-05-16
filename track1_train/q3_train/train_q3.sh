#!/bin/bash

# 全局基础路径与输出目录配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR=os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ROOT_OUTPUT_DIR="all_combinations_results_q5"
MODEL_SAVE_DIR="saved_models"

# 物理上分离的训练与验证集 CSV 路径
TRAIN_CSV=os.path.join(BASE_DIR, "train_data.csv")
VAL_CSV=os.path.join(BASE_DIR, "val_data.csv")
QTYPE="q3"

# 调用 Python 脚本
python3 run_experiment.py \
    "$BASE_DIR" \
    "$ROOT_OUTPUT_DIR" \
    "$MODEL_SAVE_DIR" \
    "$TRAIN_CSV" \
    "$VAL_CSV" \
    "$QTYPE"