# 1：测试集特征目录 (test_feature_*)：这些目录存放了预先提取好的 .npz 格式测试集特征
        
    test_feature_q1_t: 维度 q1 (general_1) 的文本模态特征。
    test_feature_q1_a: 维度 q1 (general_1) 的音频模态特征。
    test_feature_q1_v: 维度 q1 (general_1) 的视频模态特征。
    test_feature_q2_t: 维度 q2 (general_2) 的文本模态特征。
    test_feature_q2_a: 维度 q2 (general_2) 的音频模态特征。
    test_feature_q2_v: 维度 q2 (general_2) 的视频模态特征。
    test_feature_q3_t: 维度 q3 (Honesty-Humility) 的文本模态特征。
    test_feature_q3_a: 维度 q3 (Honesty-Humility) 的音频模态特征。
    test_feature_q3_v: 维度 q3 (Honesty-Humility) 的视频模态特征。
    test_feature_q4_t: 维度 q4 (Extraversion) 的文本模态特征。
    test_feature_q4_a: 维度 q4 (Extraversion) 的音频模态特征。
    test_feature_q4_v: 维度 q4 (Extraversion) 的视频模态特征。
    test_feature_q5_t: 维度 q5 (Agreeableness) 的文本模态特征。
    test_feature_q5_a: 维度 q5 (Agreeableness) 的音频模态特征。
    test_feature_q5_v: 维度 q5 (Agreeableness) 的视频模态特征。
    test_feature_q6_t: 维度 q6 (Conscientiousness) 的文本模态特征。
    test_feature_q6_a: 维度 q6 (Conscientiousness) 的音频模态特征。
    test_feature_q6_v: 维度 q6 (Conscientiousness) 的视频模态特征。
注明：q1和q2通用问题的特征仅用于track2，不涉及track1，track1只使用q3-q6特征

# 2：track1_train(track1训练代码目录)
    data：存放数据，包括原始视频，转录的音频和文本
    gemini_embedding:存放各个维度各个模态的特征
    preprocess：数据预处理代码，包括视频转音频、文本转录、特征提取
    q*_train:各个维度的训练代码
    其余csv均为数据集文件

# 3:track2_train(track2训练代码目录)

# 3：模型权重目录 (checkpoints)
    checkpoints/: 存放track1训练完成后保存的四个维度的模型权重文件 (.pth)。脚本会自动根据权重文件加载对应的模型参数。
# 4：数据模板
    template.csv: 初始的提交模板文件，包含样本 id 列，用于引导推理流程并确定样本顺序。
# 5：脚本说明
    test.py: track1测试脚本。支持模型结构自适应、模态组合选择。
    test.sh: track1启动脚本。该脚本配置了所有特征路径与权重路径参数，按顺序一次性执行 q3 → q4 → q5 → q6 的全流程推理，并将结果合并汇总导出为submission.csv。
    test_task2.py: track2测试脚本
    test_task2.sh: track2启动脚本
# 运行启动脚本:
    # 赋予执行权限
    chmod +x test.sh
    chmod +x test_task2.sh
    # 运行推理流程
    ./test.sh
    ./test_task2.sh
