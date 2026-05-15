# 目录结构说明
测试集特征目录 (test_feature_*)：这些目录存放了预先提取好的 .npz 格式测试集特征，每个维度（Question）均包含文本 (t)、音频 (a)、视频 (v) 三种模态：

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

# 模型权重目录 (checkpoints)
        checkpoints/: 存放训练完成后保存的四个维度的模型权重文件 (.pth)。脚本会自动根据权重文件加载对应的模型参数。

# 数据模板
        template.csv: 初始的提交模板文件，包含样本 id 列，用于引导推理流程并确定样本顺序。

# 脚本说明
        test.py: 测试脚本。支持模型结构自适应、模态组合选择。
        test.sh: 启动集成脚本。该脚本配置了所有特征路径与权重路径参数，按顺序一次性执行 q3 → q4 → q5 → q6 的全流程推理，并将结果汇总导出。

# 运行启动脚本:
        # 赋予执行权限
        chmod +x test.sh

        # 运行推理流程
        ./test.sh
