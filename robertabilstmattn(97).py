# -*- coding: utf-8 -*-
"""
中文评论二分类：RoBERTa + BiLSTM + Attention
特点：
1. 适配你的数据列：评论 / lable
2. 自动尝试多种编码读取CSV，解决中文乱码
3. 不做全量去重，保留真实重复评论
4. 按“评论文本”分组切分 train/val/test，避免同文本泄漏到不同集合
5. 自动移除“同一文本对应多个标签”的冲突样本
6. 类别不平衡处理：WeightedRandomSampler + FocalLoss
7. 以少数类（差评类）F1作为模型选择标准
"""

import os
import re
import math
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# =========================
# 1. 全局配置
# =========================
SEED = 42
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
DATA_PATH = "huizong_cleaned.csv"   # 改成你的文件路径
TEXT_COL = "评论"
LABEL_COL = "lable"

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 1e-2
PATIENCE = 2
NUM_WORKERS = 0  # Windows 下建议 0，更稳
DROPOUT = 0.3
LSTM_HIDDEN = 256
LSTM_LAYERS = 1

TEST_SIZE = 0.20
VAL_SIZE_IN_REMAIN = 0.125   # 先切20%测试，再从剩余80%中切12.5%，最终约等于70/10/20
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

# 是否保留重复样本：
# True  = 保留重复评论，但按文本分组切分，防止同文本泄漏
# False = 先按文本去重后再训练
KEEP_DUPLICATES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# 3. 读取数据（自动处理编码）
# =========================
def load_csv_auto_encoding(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] 成功使用编码读取数据: {enc}")
            return df, enc
        except Exception as e:
            last_error = e
    raise ValueError(f"无法读取CSV，请检查文件编码。最后错误：{last_error}")


# =========================
# 4. 文本预处理
# =========================
def clean_text(text: str) -> str:
    """
    中文任务不要做英文式强清洗，避免破坏语义。
    这里只做轻量清洗：
    1. 转字符串
    2. 去前后空格
    3. 合并多余空白
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# 5. 数据检查与清洗
# =========================
def prepare_dataframe(df, text_col, label_col, keep_duplicates=True):
    assert text_col in df.columns, f"找不到文本列: {text_col}"
    assert label_col in df.columns, f"找不到标签列: {label_col}"

    df = df[[text_col, label_col]].copy()
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    # 文本清洗
    df[text_col] = df[text_col].astype(str).apply(clean_text)

    # 标签转int
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)

    # 只保留二分类标签 0/1
    df = df[df[label_col].isin([0, 1])].reset_index(drop=True)

    # 去掉空文本
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)

    print("\n[INFO] 原始清洗后数据量:", len(df))
    print("[INFO] 标签分布:\n", df[label_col].value_counts())

    # 找出“同一文本多个标签”的冲突样本
    nunique_per_text = df.groupby(text_col)[label_col].nunique()
    conflict_texts = nunique_per_text[nunique_per_text > 1].index.tolist()

    if len(conflict_texts) > 0:
        print(f"\n[WARN] 发现 {len(conflict_texts)} 条文本存在多标签冲突，已删除这些冲突样本。")
        df = df[~df[text_col].isin(conflict_texts)].reset_index(drop=True)

    if keep_duplicates:
        print("\n[INFO] 当前策略：保留重复评论，但按文本分组切分，避免数据泄漏。")
    else:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        print(f"\n[INFO] 当前策略：先去重。删除重复后样本数：{len(df)}（删除了 {before - len(df)} 条）")

    print("\n[INFO] 最终可用数据量:", len(df))
    print("[INFO] 最终标签分布:\n", df[label_col].value_counts())
    return df


# =========================
# 6. 分组切分（关键）
# =========================
def grouped_train_val_test_split(df, text_col, label_col, test_size=0.2, val_size_in_remain=0.125, random_state=42):
    """
    核心思想：
    - 先按文本分组，每个唯一文本只属于一个group
    - 对group做分层切分
    - 再映射回原始样本
    这样保留重复样本，但不会让同文本泄漏到不同集合
    """
    group_df = df.groupby(text_col, as_index=False)[label_col].first()

    trainval_groups, test_groups = train_test_split(
        group_df,
        test_size=test_size,
        random_state=random_state,
        stratify=group_df[label_col]
    )

    train_groups, val_groups = train_test_split(
        trainval_groups,
        test_size=val_size_in_remain,
        random_state=random_state,
        stratify=trainval_groups[label_col]
    )

    train_texts = set(train_groups[text_col].tolist())
    val_texts = set(val_groups[text_col].tolist())
    test_texts = set(test_groups[text_col].tolist())

    train_df = df[df[text_col].isin(train_texts)].reset_index(drop=True)
    val_df = df[df[text_col].isin(val_texts)].reset_index(drop=True)
    test_df = df[df[text_col].isin(test_texts)].reset_index(drop=True)

    # 检查是否泄漏
    assert len(train_texts & val_texts) == 0
    assert len(train_texts & test_texts) == 0
    assert len(val_texts & test_texts) == 0

    print("\n[INFO] 数据集划分结果（按文本分组，无泄漏）")
    print(f"Train: {len(train_df)}")
    print(train_df[label_col].value_counts())
    print(f"\nVal: {len(val_df)}")
    print(val_df[label_col].value_counts())
    print(f"\nTest: {len(test_df)}")
    print(test_df[label_col].value_counts())

    return train_df, val_df, test_df


# =========================
# 7. Dataset
# =========================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        return item


# =========================
# 8. 模型
# =========================
class RobertaBiLSTMAttn(nn.Module):
    def __init__(self, model_name, num_labels=2, lstm_hidden=256, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.attn_fc = nn.Linear(lstm_hidden * 2, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        seq_emb = outputs.last_hidden_state  # [B, L, H]

        lstm_out, _ = self.lstm(seq_emb)     # [B, L, 2H]

        attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # [B, L]
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, L, 1]

        pooled = torch.sum(lstm_out * attn_weights, dim=1)  # [B, 2H]
        logits = self.classifier(pooled)
        return logits


# =========================
# 9. Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)          # [B]
        pt = torch.exp(-ce_loss)                    # [B]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# =========================
# 10. 指标计算
# =========================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 少数类（默认按你的数据，0是差评少数类）
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    result = {
        "acc": acc,
        "weighted_precision": precision_w,
        "weighted_recall": recall_w,
        "weighted_f1": f1_w,
        "macro_precision": precision_m,
        "macro_recall": recall_m,
        "macro_f1": f1_m,
        "minority_precision": precision_per_class[0],
        "minority_recall": recall_per_class[0],
        "minority_f1": f1_per_class[0],
        "minority_support": support_per_class[0]
    }
    return result


# =========================
# 11. 训练与评估
# =========================
def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="训练中", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="评估中", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


# =========================
# 12. 构造WeightedRandomSampler
# =========================
def build_weighted_sampler(labels):
    labels = np.array(labels)
    class_count = np.bincount(labels)
    class_weights = 1.0 / class_count
    sample_weights = class_weights[labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# =========================
# 13. 主流程
# =========================
def main():
    print("=" * 80)
    print("设备:", DEVICE)
    print("=" * 80)

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    # 读取数据
    df, used_encoding = load_csv_auto_encoding(DATA_PATH)

    # 清洗数据
    df = prepare_dataframe(
        df=df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        keep_duplicates=KEEP_DUPLICATES
    )

    # 划分 train / val / test
    train_df, val_df, test_df = grouped_train_val_test_split(
        df=df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        test_size=TEST_SIZE,
        val_size_in_remain=VAL_SIZE_IN_REMAIN,
        random_state=SEED
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # dataset
    train_dataset = ReviewDataset(
        texts=train_df[TEXT_COL].tolist(),
        labels=train_df[LABEL_COL].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_dataset = ReviewDataset(
        texts=val_df[TEXT_COL].tolist(),
        labels=val_df[LABEL_COL].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_dataset = ReviewDataset(
        texts=test_df[TEXT_COL].tolist(),
        labels=test_df[LABEL_COL].tolist(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    # sampler：只对训练集做
    train_sampler = build_weighted_sampler(train_df[LABEL_COL].tolist())

    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # 模型
    model = RobertaBiLSTMAttn(
        model_name=MODEL_NAME,
        num_labels=2,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    # 只在训练集上计算类权重
    y_train = train_df[LABEL_COL].to_numpy()
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print("\n[INFO] 训练集 class weights:", class_weights.tolist())

    # loss
    if USE_FOCAL_LOSS:
        loss_fn = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
        print("[INFO] 当前损失函数: FocalLoss")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print("[INFO] 当前损失函数: 加权 CrossEntropyLoss")

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_metric = -1
    best_epoch = 0
    patience_counter = 0
    best_model_path = "best_model_chinese_roberta_bilstm_attn.pt"

    # 训练
    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 80)

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, DEVICE
        )
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, loss_fn, DEVICE
        )

        print(f"[Train] loss={train_loss:.4f} acc={train_metrics['acc']:.4f} "
              f"macro_f1={train_metrics['macro_f1']:.4f} minority_f1={train_metrics['minority_f1']:.4f}")

        print(f"[Val]   loss={val_loss:.4f} acc={val_metrics['acc']:.4f} "
              f"macro_f1={val_metrics['macro_f1']:.4f} minority_f1={val_metrics['minority_f1']:.4f} "
              f"minority_recall={val_metrics['minority_recall']:.4f}")

        # 关键：按少数类F1选最好模型
        current_metric = val_metrics["minority_f1"]

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "max_len": MAX_LEN,
                "used_encoding": used_encoding,
                "keep_duplicates": KEEP_DUPLICATES
            }, best_model_path)

            print(f"[INFO] 已保存最佳模型到: {best_model_path}")
        else:
            patience_counter += 1
            print(f"[INFO] 验证集少数类F1未提升，patience = {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("[INFO] 触发早停。")
            break

    print("\n" + "=" * 80)
    print(f"训练结束，最佳 epoch = {best_epoch}, 最佳验证集 minority_f1 = {best_metric:.4f}")
    print("=" * 80)

    # 加载最佳模型
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 测试集评估
    test_loss, test_metrics, y_true, y_pred = evaluate(
        model, test_loader, loss_fn, DEVICE
    )

    print("\n" + "=" * 80)
    print("测试集结果")
    print("=" * 80)
    print(f"Test loss           : {test_loss:.4f}")
    print(f"Test accuracy       : {test_metrics['acc']:.4f}")
    print(f"Test weighted_f1    : {test_metrics['weighted_f1']:.4f}")
    print(f"Test macro_f1       : {test_metrics['macro_f1']:.4f}")
    print(f"Test minority_f1    : {test_metrics['minority_f1']:.4f}")
    print(f"Test minority_recall: {test_metrics['minority_recall']:.4f}")

    print("\n分类报告（0=差评少数类, 1=多数类）：")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("混淆矩阵：")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\n[INFO] 说明：")
    print("1. 这版代码默认保留重复评论，但按文本分组切分，避免泄漏。")
    print("2. 如果你想改成先去重再训练，把 KEEP_DUPLICATES = False 即可。")
    print("3. 对你这种差评少的任务，不要只看 accuracy，重点看 minority_f1 和 minority_recall。")


if __name__ == "__main__":
    main()