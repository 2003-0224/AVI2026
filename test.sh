#!/bin/bash

# 基础路径配置
TEST_CSV="./template.csv"
OUTPUT_CSV="./submission.csv"

# ====================================================================
# 配置 4 个维度的模型权重路径 (请根据实际单模态、多模态文件名进行替换)
# ====================================================================
MODEL_Q3="./checkpoints/model_q3_t_a.pth"
MODEL_Q4="./checkpoints/model_q4_t_a.pth"
MODEL_Q5="./checkpoints/model_q5_t.pth"
MODEL_Q6="./checkpoints/model_q6_t_a_v.pth"

# ====================================================================
# 配置每个问题对应的各模态特征根目录路径
# ====================================================================
Q3_T_DIR="./test_feature_q3_t"
Q3_A_DIR="./test_feature_q3_a"
Q3_V_DIR="./test_feature_q3_v"

Q4_T_DIR="./test_feature_q4_t"
Q4_A_DIR="./test_feature_q4_a"
Q4_V_DIR="./test_feature_q4_v"

Q5_T_DIR="./test_feature_q5_t"
Q5_A_DIR="./test_feature_q5_a"
Q5_V_DIR="./test_feature_q5_v"

Q6_T_DIR="./test_feature_q6_t"
Q6_A_DIR="./test_feature_q6_a"
Q6_V_DIR="./test_feature_q6_v"

# 执行多任务串行推理
python test.py \
    --test_data_path "$TEST_CSV" \
    --output_result_path "$OUTPUT_CSV" \
    --model_q3 "$MODEL_Q3" \
    --model_q4 "$MODEL_Q4" \
    --model_q5 "$MODEL_Q5" \
    --model_q6 "$MODEL_Q6" \
    --q3_t_dir "$Q3_T_DIR" --q3_a_dir "$Q3_A_DIR" --q3_v_dir "$Q3_V_DIR" \
    --q4_t_dir "$Q4_T_DIR" --q4_a_dir "$Q4_A_DIR" --q4_v_dir "$Q4_V_DIR" \
    --q5_t_dir "$Q5_T_DIR" --q5_a_dir "$Q5_A_DIR" --q5_v_dir "$Q5_V_DIR" \
    --q6_t_dir "$Q6_T_DIR" --q6_a_dir "$Q6_A_DIR" --q6_v_dir "$Q6_V_DIR"
