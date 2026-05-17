import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate_fn_test(batch):
    # batch: List of (features_dict, sample_id)
    features_list = [item[0] for item in batch]
    sample_ids = [item[1] for item in batch]

    features = {}
    masks = {}

    for k in features_list[0].keys():
        modality_tensors = [f[k] for f in features_list]

        if k == 'audio':
            lengths = [t.shape[0] for t in modality_tensors]
            max_len = max(lengths)
            padded = pad_sequence(modality_tensors, batch_first=True)  # (B, T, D)
            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            audio_mask = mask.float()
            features[k] = padded
            masks[k + "_mask"] = audio_mask
        else:
            features[k] = torch.stack(modality_tensors)

    return features, masks, sample_ids

def collate_fn_train(batch):
    # batch: List of (features_dict, sample_id)
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    features = {}
    masks = {}

    for k in features_list[0].keys():
        modality_tensors = [f[k] for f in features_list]

        if k == 'audio':
            lengths = [t.shape[0] for t in modality_tensors]
            max_len = max(lengths)
            padded = pad_sequence(modality_tensors, batch_first=True)  # (B, T, D)
            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            audio_mask = mask.float()
            features[k] = padded
            masks[k + "_mask"] = audio_mask
        else:
            features[k] = torch.stack(modality_tensors)

    return features, masks, labels

class MultimodalDatasetForTrainT1(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, label_col, args=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir   # Directory containing audio features
        self.video_dir = video_dir   # Directory containing video features
        self.text_dir = text_dir     # Directory containing text features
        self.question = question     
        self.label_col = label_col
        self.training_modal = args.modalities if args else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{self.question}")]
        video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{self.question}")]
        text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{self.question}")]


        if len(audio_file) == 0 or len(video_file) == 0 or len(text_file) == 0:
            raise FileNotFoundError(f"Files for {sample_id}_{self.question} not found.")

        features = {}

        if 'audio' in self.training_modal:
            features['audio'] = np.load(os.path.join(self.audio_dir, audio_file[0]))
            # print(f"audio.shape: {features['audio'].shape}")  # 例如 (T, D_a)

        if 'video' in self.training_modal:
            features['video'] = np.load(os.path.join(self.video_dir, video_file[0]))
            # print(f"video.shape: {features['video'].shape}")  # 例如 (D_v,)

        if 'text' in self.training_modal:
            features['text'] = np.load(os.path.join(self.text_dir, text_file[0]))
            # print(f"text.shape: {features['text'].shape}")  # 例如 (D_t,)


        # label这里的归一化，把1-5的label映射到0-1
        label = self.data.iloc[idx][self.label_col]
        label_normalized = (label - 1) / 4

        # 返回的tensor改成列表，分别返回各模态tensor，而不预先concatenate
        # 返回字典形式的特征 + 标签
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}, \
                torch.tensor(label_normalized, dtype=torch.float32)
    
class MultimodalDatasetForTestT1(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, args=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.question = question
        self.test_modal = args.modalities if args else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{self.question}")]
        video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{self.question}")]
        text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{self.question}")]

        # if len(audio_file) == 0 or len(video_file) == 0 or len(text_file) == 0:
        #     raise FileNotFoundError(f"Files for {sample_id}_{self.question} not found.")


        features = {}  # 使用字典存储各模态特征（无需对齐）

        if 'audio' in self.test_modal:
            features['audio'] = np.load(os.path.join(self.audio_dir, audio_file[0]))
            # print(f"audio.shape: {features['audio'].shape}")  # 例如 (T, D_a)

        if 'video' in self.test_modal:
            features['video'] = np.load(os.path.join(self.video_dir, video_file[0]))
            # print(f"video.shape: {features['video'].shape}")  # 例如 (D_v,)

        if 'text' in self.test_modal:
            features['text'] = np.load(os.path.join(self.text_dir, text_file[0]))
            # print(f"text.shape: {features['text'].shape}")  # 例如 (D_t,)

        return {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}, sample_id