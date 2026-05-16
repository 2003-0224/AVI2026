import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AttentionFusionRegressor(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dims, dropout_rate, num_modalities=3):
        super().__init__()

        # 第一层：跨模态自注意力
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(input_dim)
        # 一个可学习的查询向量
        self.query_vector = nn.Parameter(torch.zeros(1, 1, input_dim))
        # 用于池化的第二层注意力
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.0,  # 池化层通常不加dropout
            batch_first=True
        )
        self.regressor = RegressionHead(input_dim, hidden_dims, dropout_rate)

    def forward(self, x):
        # 1. 跨模态交互
        attn_output, _ = self.cross_modal_attention(x, x, x)
        fused_seq = self.ln1(x + attn_output)
        # === 2. 使用注意力池化 ===
        batch_size = x.shape[0]
        query = self.query_vector.expand(batch_size, -1, -1)
        # 使用查询向量作为Query，交互后的序列作为Key和Value
        # query去“查询”fused_seq中的信息
        fused_vector, _ = self.pooling_attention(query, fused_seq, fused_seq)
        fused_vector = fused_vector.squeeze(1)
        # 3. 送入回归头
        return self.regressor(fused_vector)