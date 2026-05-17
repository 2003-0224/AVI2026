import torch


class GPT2Shared(torch.nn.Module):
    def __init__(self, args):
        super(GPT2Shared, self).__init__()
        self.args = args
        self.base_dim = 1536

        self.video_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.video_dim, self.base_dim * 3),
            torch.nn.GELU(),
            torch.nn.Linear(self.base_dim * 3, self.base_dim)
        )

        self.audio_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.audio_dim, self.base_dim * 3),
            torch.nn.GELU(),
            torch.nn.Linear(self.base_dim * 3, self.base_dim)
        )

        self.text_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.text_dim, self.base_dim * 3),
            torch.nn.GELU(),
            torch.nn.Linear(self.base_dim * 3, self.base_dim)
        )

        self.metadata_dim = getattr(args, "metadata_dim", 0)
        self.metadata_hidden_dim = 64 if self.metadata_dim > 0 else 0
        if self.metadata_dim > 0:
            self.metadata_adapter = torch.nn.Sequential(
                torch.nn.Linear(self.metadata_dim, self.metadata_hidden_dim),
                torch.nn.LayerNorm(self.metadata_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2)
            )
        else:
            self.metadata_adapter = None
        
        self.attention = torch.nn.MultiheadAttention(embed_dim=self.base_dim, num_heads=8, batch_first=True)
        classifier_input_dim = self.base_dim + self.metadata_hidden_dim

        self.ensemble = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(classifier_input_dim, self.base_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.base_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, args.target_dim)
            ) for _ in range(32)
        ])

    def forward(self, audio_feat, video_feat, text_feat, metadata_feat=None):
        # Get batch size, sequence length and feature dimension
        B, T, _ = video_feat.shape
        video_feat = video_feat.reshape(B * T, -1)
        text_feat = text_feat.reshape(B * T, -1)
        audio_feat = audio_feat.reshape(B * T, -1)
        # Project features through adapters and feature projection
        video_feat = self.video_adapter(video_feat)  # [B*T, D]
        audio_feat = self.audio_adapter(audio_feat)  # [B*T, D]
        text_feat = self.text_adapter(text_feat)  # [B*T, D]
  
        multi_modal_chunk = torch.stack([video_feat, text_feat, audio_feat], dim=1)  # [B*T, 3, D]
        multi_modal_chunk = self.attention(query=text_feat.unsqueeze(1),
                                           key=multi_modal_chunk,
                                           value=multi_modal_chunk)[0]
        if self.metadata_adapter is not None:
            if metadata_feat is None:
                metadata_feat = torch.zeros(B, self.metadata_dim, device=multi_modal_chunk.device)
            metadata_feat = self.metadata_adapter(metadata_feat)
            metadata_feat = metadata_feat.unsqueeze(1).expand(-1, T, -1).reshape(B * T, 1, -1)
            multi_modal_chunk = torch.cat([multi_modal_chunk, metadata_feat], dim=-1)
        outputs = torch.stack([mlp(multi_modal_chunk) for mlp in self.ensemble], dim=0) # [32, B*T, target_dim]
        logits = outputs.mean(dim=0)  # [B*T, target_dim]
        logits = logits.view(B, T, -1).mean(dim=1)  # [B, target_dim]
        return logits
