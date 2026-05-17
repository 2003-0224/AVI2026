import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import ast

def _question_level_feature(path):
    feature = np.load(path)
    if isinstance(feature, np.lib.npyio.NpzFile):
        if "embedding" in feature.files:
            feature = feature["embedding"]
        elif len(feature.files) == 1:
            feature = feature[feature.files[0]]
        else:
            raise ValueError(f"Cannot choose array from {path}; npz keys={feature.files}")
    if feature.ndim == 1:
        return feature
    return feature.reshape(-1, feature.shape[-1]).mean(axis=0)


def _metadata_feature(row, metadata_cols):
    values = []
    for col in metadata_cols:
        if col == "gender":
            value = float(row[col]) / 4.0 if col in row and pd.notna(row[col]) else 0.5
        elif col == "age":
            value = float(row[col]) / 100.0 if col in row and pd.notna(row[col]) else 0.35
        elif col == "education":
            value = float(row[col]) / 7.0 if col in row and pd.notna(row[col]) else 0.5
        elif col == "work_experience":
            value = float(row[col]) / 60.0 if col in row and pd.notna(row[col]) else 0.25
        elif col in {"H_self", "E_self", "A_self", "C_self"}:
            value = (float(row[col]) - 1.0) / 4.0 if col in row and pd.notna(row[col]) else 0.5
        else:
            value = float(row[col]) if col in row and pd.notna(row[col]) else 0.0
        values.append(value)
    return np.array(values, dtype=np.float32)


class MultimodalDatasetForTrainT2(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, label_col, rating_csv, args=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir  # Directory containing audio features
        self.video_dir = video_dir  # Directory containing video features
        self.text_dir = text_dir    # Directory containing text features
        self.question = question
        self.training_modal = args.modalities if args else None
        self.metadata_cols = args.metadata_cols if args else []
        if isinstance(label_col, list):
            if len(label_col) != 1:
                raise ValueError("Track2 2026 is single-label classification. Use --label_col g_level.")
            self.label_col = label_col[0]
        else:
            self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']

        audio_files = []
        video_files = []
        text_files = []
        features = {}

        for q in self.question:
            audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")]
            video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")]
            text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")]
            missing = []
            if not audio_file:
                missing.append(f"audio in {self.audio_dir}")
            if not video_file:
                missing.append(f"video in {self.video_dir}")
            if not text_file:
                missing.append(f"text in {self.text_dir}")
            if missing:
                raise FileNotFoundError(f"Missing {'; '.join(missing)} for {sample_id}_{q}")
            audio_files.append(audio_file[0])
            video_files.append(video_file[0])
            text_files.append(text_file[0])

        # 每个问题聚合成一个向量，q1-q6 堆叠为面试级序列。
        if 'audio' in self.training_modal:
            features['audio'] = np.stack([_question_level_feature(os.path.join(self.audio_dir, f)) for f in audio_files], axis=0)
        
        if 'video' in self.training_modal:
            features['video'] = np.stack([_question_level_feature(os.path.join(self.video_dir, f)) for f in video_files], axis=0)

        if 'text' in self.training_modal:
            features['text'] = np.stack([_question_level_feature(os.path.join(self.text_dir, f)) for f in text_files], axis=0)

        if self.metadata_cols:
            features['metadata'] = _metadata_feature(self.data.iloc[idx], self.metadata_cols)

        label = int(self.data.iloc[idx][self.label_col]) - 1

        return {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}, torch.tensor(label, dtype=torch.long)

class MultimodalDatasetForTestT2(Dataset):
    def __init__(self, csv_file, audio_dir, video_dir, text_dir, question, rating_csv, args=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        if isinstance(question, list) and len(question) == 1 and isinstance(question[0], str):
            # 将字符串形式的列表变成真正的 Python 列表
            self.question = ast.literal_eval(question[0])
        else:
            self.question = question
        self.training_modal = args.modalities if args else None
        self.metadata_cols = args.metadata_cols if args else []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']
        audio_files, video_files, text_files = [], [], []
        features = {}

        for q in self.question:
            audio_file = [f for f in os.listdir(self.audio_dir) if f.startswith(f"{sample_id}_{q}")]
            video_file = [f for f in os.listdir(self.video_dir) if f.startswith(f"{sample_id}_{q}")]
            text_file = [f for f in os.listdir(self.text_dir) if f.startswith(f"{sample_id}_{q}")]

            missing = []
            if not audio_file:
                missing.append(f"audio in {self.audio_dir}")
            if not video_file:
                missing.append(f"video in {self.video_dir}")
            if not text_file:
                missing.append(f"text in {self.text_dir}")
            if missing:
                raise FileNotFoundError(f"Missing {'; '.join(missing)} for {sample_id}_{q}")
            audio_files.append(audio_file[0])
            video_files.append(video_file[0])
            text_files.append(text_file[0])
        
        if 'audio' in self.training_modal:
            features['audio'] = np.stack([_question_level_feature(os.path.join(self.audio_dir, f)) for f in audio_files], axis=0)
        
        if 'video' in self.training_modal:
            features['video'] = np.stack([_question_level_feature(os.path.join(self.video_dir, f)) for f in video_files], axis=0)

        if 'text' in self.training_modal:
            features['text'] = np.stack([_question_level_feature(os.path.join(self.text_dir, f)) for f in text_files], axis=0)

        if self.metadata_cols:
            features['metadata'] = _metadata_feature(self.data.iloc[idx], self.metadata_cols)

        return {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}, sample_id
    



def collate_fn_train(batch):
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    features = {}
    masks = {}
    for k in features_list[0].keys():
        modality_tensors = [f[k] for f in features_list]

        if k == 'audio':
            modality_tensors = [item[0][k] for item in batch]
            lengths = [t.shape[0] for t in modality_tensors]
            max_len = max(lengths)
            padded = pad_sequence(modality_tensors, batch_first=True)
            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            audio_mask = mask.float()
            features[k] = padded
            masks[k + "_mask"] = audio_mask
        else:
            features[k] = torch.stack(modality_tensors)

    return features, masks, labels


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
