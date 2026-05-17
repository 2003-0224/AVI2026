import os
import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from baseline_dataset2_vote import MultimodalDatasetForTrainT2, MultimodalDatasetForTestT2
from baseline_dataset2_vote import collate_fn_train, collate_fn_test
from tqdm import tqdm, trange
from M_model_T import GPT2Shared
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for features, mask, labels in train_bar:
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features['audio'], features['video'], features['text'], features.get('metadata'))
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)

def evaluate_model(model, loader, criterion, device, is_test=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, mask, labels in loader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            outputs = model(features['audio'], features['video'], features['text'], features.get('metadata'))
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(criterion.weight.numel())) if criterion.weight is not None else None)
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
    }

def compute_class_weights(dataset, num_classes, device):
    labels = dataset.data[dataset.label_col].astype(int).to_numpy() - 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"Class counts: {counts.astype(int).tolist()} | class weights: {weights.detach().cpu().numpy().round(4).tolist()}")
    return weights

def test_model(model, test_loader, device, output_csv_path, test_csv_path):
    model.eval()
    predictions = []
    ids_list = []
    
    with torch.no_grad():
        for features, mask, ids in test_loader:
            features = {k: v.to(device) for k, v in features.items()}
            outputs = model(features['audio'], features['video'], features['text'], features.get('metadata'))
            preds = torch.argmax(outputs, dim=1) + 1
            predictions.append(preds.detach().cpu().numpy())
            ids_list.extend(ids)
    
    all_preds = np.concatenate(predictions)
    result_df = pd.DataFrame({"g_level": all_preds.astype(int)})
    result_df.insert(0, "id", ids_list)
    result_df.to_csv(output_csv_path, index=False)
    print(f"✅ Predictions saved to {output_csv_path}")

def fit_model(args, train_set, val_set, device):
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              collate_fn=collate_fn_train, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            collate_fn=collate_fn_train,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = GPT2Shared(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    class_weights = compute_class_weights(train_set, args.target_dim, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_macro_f1 = -1.0
    best_metrics = None
    best_model = None

    for epoch in trange(args.num_epochs, desc="Epochs"):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics['macro_f1'])

        if val_metrics['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics['macro_f1']
            best_metrics = val_metrics
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    model.load_state_dict(best_model)
    torch.save(best_model, args.output_model)
    print(f"🏆 Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(
        f"Best Val Accuracy: {best_metrics['accuracy']:.4f} | "
        f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Best Val Confusion Matrix:\n{best_metrics['confusion_matrix']}")
    print(f"✅ Best model saved to {args.output_model}")
    return model

def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve saved to {save_path}")

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser()
    # 数据集参数
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--label_col', nargs='+', required=True)
    parser.add_argument('--question', nargs='+', required=True)
    parser.add_argument('--rating_csv', default=None)
    parser.add_argument('--metadata_cols', nargs='+',
                        default=['gender', 'age', 'education', 'work_experience',
                                 'H_self', 'E_self', 'A_self', 'C_self'])

    # 输入特征参数
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--text_dir', required=True)
    parser.add_argument('--audio_dim', type=int, default=384)
    parser.add_argument('--video_dim', type=int, default=512)
    parser.add_argument('--text_dim', type=int, default=768)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--optim', type=str, default='adamw')

    # 测试参数
    parser.add_argument('--only_test', action='store_true', default=False)
    parser.add_argument('--test_output_csv', type=str, default='test_predictions.csv')
    parser.add_argument('--test_model', default='best_model.pth')

    # 模型参数
    parser.add_argument('--HCPdropout_audio', type=float, default=0.2)
    parser.add_argument('--HCPdropout_video', type=float, default=0.2)
    parser.add_argument('--HCPdropout_text', type=float, default=0.2)
    parser.add_argument('--HCPdropout_pure_text', type=float, default=0.1)
    parser.add_argument('--use_prompt', type=bool, default=False)
    parser.add_argument('--unified_dim', type=int, default=512)
    parser.add_argument('--heads_num', type=int, default=4)
    parser.add_argument('--ATCdropout', type=float, default=0.3)
    parser.add_argument('--VTCdropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--enhancer_dim', type=int, default=512)
    parser.add_argument('--TFEdropout', type=float, default=0.2)
    parser.add_argument('--RHdropout', type=float, default=0.2)
    parser.add_argument('--target_dim', type=int, default=3)
    parser.add_argument('--num_modalities', type=int, default=3)
    parser.add_argument('--modalities', type=str, default="audio,video,text")
    parser.add_argument('--output_model', default='best_model.pth')
    parser.add_argument('--loss_plot_path', type=str, default='loss_plot.png')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--training_time', type=str)
    args = parser.parse_args()
    args.metadata_dim = len(args.metadata_cols)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args_file = os.path.join(args.log_dir, f"args_{timestamp}.json")
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"📝 Args saved to {args_file}")
    
    if args.label_col != ['g_level']:
        raise ValueError("Track2 2026 should use --label_col g_level")

    train_set = MultimodalDatasetForTrainT2(
        args.train_csv, args.audio_dir, args.video_dir, 
        args.text_dir, args.question, args.label_col, 
        args.rating_csv, args
    )
    val_set = MultimodalDatasetForTrainT2(
        args.val_csv, args.audio_dir, args.video_dir,
        args.text_dir, args.question, args.label_col,
        args.rating_csv, args
    )
    test_set = MultimodalDatasetForTestT2(
        args.test_csv, args.audio_dir, args.video_dir, 
        args.text_dir, args.question, args.rating_csv, args
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                            collate_fn=collate_fn_test,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    if not args.only_test:
        model = fit_model(args, train_set, val_set, device)
        
        print("Generating predictions on the test set...")
        test_model(model, test_loader, device, args.test_output_csv, args.test_csv)
    else:
        model = GPT2Shared(args).to(device)
        model.load_state_dict(torch.load(args.test_model, map_location=device))
        test_model(model, test_loader, device, args.test_output_csv, args.test_csv)

if __name__ == '__main__':
    main()
