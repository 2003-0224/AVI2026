import torch
import torch.nn as nn


class MLPRegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


class TextCenteredCrossModalAttentionRegressor(nn.Module):
    """
    Text-centered one-way cross-modal attention.
    Text is used as Query; Audio and Video are used as Key/Value.
    """
    def __init__(
        self,
        embed_dim=1536,
        attn_dim=512,
        num_heads=8,
        hidden_dims=(256, 64),
        dropout_rate=0.1,
    ):
        super().__init__()

        self.text_proj = nn.Linear(embed_dim, attn_dim)
        self.audio_proj = nn.Linear(embed_dim, attn_dim)
        self.video_proj = nn.Linear(embed_dim, attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = MLPRegressionHead(
            in_dim=attn_dim * 2,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )

    def forward(self, text, audio, video):
        q = self.text_proj(text).unsqueeze(1)         
        a = self.audio_proj(audio).unsqueeze(1)     
        v = self.video_proj(video).unsqueeze(1)     
        kv = torch.cat([a, v], dim=1)           
        ctx, _ = self.attn(query=q, key=kv, value=kv) 
        ctx = self.norm(q + self.dropout(ctx)).squeeze(1)
        q = q.squeeze(1)
        fused = torch.cat([q, ctx], dim=-1)
        return self.regressor(fused)
