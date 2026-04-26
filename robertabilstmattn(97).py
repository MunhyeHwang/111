# -*- coding: utf-8 -*-
"""
RoBERTa-BiLSTM-Attention 中文评论二分类
功能：
1）使用 huizong_cleaned.csv 训练/验证模型
2）输出负面评价（差评类）Accuracy 变化图
3）将 huizong_cleaned 全量数据导入模型做预测
4）按四个维度统计好评/差评数量与好评率
5）不删除重复评论；同一评论可对应多个维度
"""

import os
import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# 手动加载中文字体
font_path = "./simhei.ttf"
CN_FONT = FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False
print(f"[INFO] 已加载字体: {font_path}")

warnings.filterwarnings("ignore")

# 1. 配置
SEED = 42
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
DATA_PATH = "huizong_cleaned.csv"
TEXT_COL = "评论"
LABEL_COL = "lable"

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 6
ENCODER_LR = 1e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.4
LSTM_HIDDEN = 256
PATIENCE = 2
FOCAL_GAMMA = 2.0
MINORITY_CLASS = 0  # 0=差评，1=好评
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH = "best_roberta_bilstm_attn.pt"
NEG_ACC_FIG_PATH = "差评准确率变化曲线.png"
ASPECT_RESULT_PATH = "四维度好差评统计.xlsx"
ASPECT_BAR_FIG_PATH = "四维度好评差评柱状图.png"
PROF_WORDCLOUD_FIG_PATH = "专业性词频气泡图.png"
SAFE_WORDCLOUD_FIG_PATH = "安全性词频气泡图.png"

# 维度关键词（按你截图整理，可继续补充）
ASPECT_KEYWORDS = {
    "安全性": [
        "干净", "无菌", "消毒", "卫生", "口罩", "健康", "包扎", "检测", "齐全", "包装",
        "感染", "规范", "交叉感染", "操作", "耐受", "过敏", "清洁", "防护", "酒精"
    ],
    "专业性": [
        "专业", "手法", "打针", "疼痛", "拆线", "注射", "能力", "询问", "规范", "检查",
        "护理", "核对", "手术", "娴熟", "采血", "不痛", "扎针", "换药", "伤口", "详细",
        "检测", "质量", "不准", "采样", "技术"
    ],
    "响应性": [
        "准时", "便捷", "时间", "快速", "准时到达", "方便快捷", "在家", "速度", "上门服务",
        "行动不便", "预约", "提前", "收到", "过程", "准时", "排队", "物流", "挺快", "快捷",
        "及时", "效率", "上门", "方便"
    ],
    "服务性": [
        "态度", "耐心", "心细", "放心", "温柔", "细致", "轻柔", "仔细", "礼貌", "亲切",
        "服务态度", "细心", "讲解", "注意事项", "服务周到", "贴心", "热情", "沟通", "性价比",
        "体验", "报告", "便宜", "认真负责", "实惠", "安心", "周到", "客服", "服务"
    ]
}

# 2. 工具函数
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_csv_auto(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] 读取成功，编码={enc}")
            return df
        except Exception as e:
            last_error = e
    raise ValueError(f"CSV读取失败: {last_error}")

def clean_text(x):
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

def prepare_dataframe(df):
    assert TEXT_COL in df.columns, f"缺少列: {TEXT_COL}"
    assert LABEL_COL in df.columns, f"缺少列: {LABEL_COL}"

    df = df[[TEXT_COL, LABEL_COL]].copy()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(clean_text)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[LABEL_COL].isin([0, 1])].reset_index(drop=True)
    df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)

    # 删除“同一文本多标签冲突”的样本，避免训练混乱
    conflict = df.groupby(TEXT_COL)[LABEL_COL].nunique()
    conflict_texts = conflict[conflict > 1].index.tolist()
    if conflict_texts:
        print(f"[WARN] 发现冲突文本 {len(conflict_texts)} 条，已删除")
        df = df[~df[TEXT_COL].isin(conflict_texts)].reset_index(drop=True)

    print(f"[INFO] 样本量: {len(df)}")
    print(df[LABEL_COL].value_counts())
    return df

def grouped_split(df, test_size=0.2, val_size_in_remain=0.125, random_state=42):
    """
    保留重复评论，但按评论文本分组切分，防止相同文本泄漏到 train/val/test
    """
    group_df = df.groupby(TEXT_COL, as_index=False)[LABEL_COL].first()

    trainval_groups, test_groups = train_test_split(
        group_df,
        test_size=test_size,
        stratify=group_df[LABEL_COL],
        random_state=random_state
    )
    train_groups, val_groups = train_test_split(
        trainval_groups,
        test_size=val_size_in_remain,
        stratify=trainval_groups[LABEL_COL],
        random_state=random_state
    )

    train_texts = set(train_groups[TEXT_COL])
    val_texts = set(val_groups[TEXT_COL])
    test_texts = set(test_groups[TEXT_COL])

    train_df = df[df[TEXT_COL].isin(train_texts)].reset_index(drop=True)
    val_df = df[df[TEXT_COL].isin(val_texts)].reset_index(drop=True)
    test_df = df[df[TEXT_COL].isin(test_texts)].reset_index(drop=True)

    print("\n[INFO] 数据划分：")
    print("Train:", len(train_df), "\n", train_df[LABEL_COL].value_counts())
    print("Val  :", len(val_df), "\n", val_df[LABEL_COL].value_counts())
    print("Test :", len(test_df), "\n", test_df[LABEL_COL].value_counts())
    return train_df, val_df, test_df

def build_weighted_sampler(labels):
    labels = np.array(labels)
    class_count = np.bincount(labels)
    class_weights = 1.0 / class_count
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_c, r_c, f1_c, s_c = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {
        "acc": acc,
        "weighted_f1": f1_w,
        "macro_f1": f1_m,
        "minority_precision": p_c[MINORITY_CLASS],
        "minority_recall": r_c[MINORITY_CLASS],
        "minority_f1": f1_c[MINORITY_CLASS],
        "minority_acc": r_c[MINORITY_CLASS],  # 对差评类来说，召回率就是“差评识别准确率”
        "minority_support": s_c[MINORITY_CLASS]
    }

def predict_with_threshold(logits, threshold=0.5):
    probs = torch.softmax(logits, dim=1)
    neg_probs = probs[:, MINORITY_CLASS]
    preds = torch.where(
        neg_probs >= threshold,
        torch.zeros_like(neg_probs, dtype=torch.long),
        torch.ones_like(neg_probs, dtype=torch.long)
    )
    return preds, neg_probs

def search_best_threshold(y_true, neg_probs, thresholds=np.arange(0.2, 0.81, 0.02)):
    y_true = np.array(y_true)
    neg_probs = np.array(neg_probs)
    best_th, best_f1, best_metrics = 0.5, -1, None
    for th in thresholds:
        pred = np.where(neg_probs >= th, 0, 1)
        metrics = compute_metrics(y_true, pred)
        if metrics["minority_f1"] > best_f1:
            best_f1 = metrics["minority_f1"]
            best_th = float(th)
            best_metrics = metrics
    return best_th, best_metrics

def freeze_encoder(encoder, freeze_embeddings=True, freeze_layers=6):
    if freeze_embeddings and hasattr(encoder, "embeddings"):
        for p in encoder.embeddings.parameters():
            p.requires_grad = False
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        for layer in encoder.encoder.layer[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

# 3. Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }

# 4. 模型
class RobertaBiLSTMAttention(nn.Module):
    def __init__(self, model_name, num_labels=2, lstm_hidden=256, dropout=0.4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            force_download=True,
            use_safetensors=False
        )
        hidden_size = self.encoder.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = out.last_hidden_state
        lstm_out, _ = self.lstm(seq)
        score = self.attn(lstm_out).squeeze(-1)
        score = score.masked_fill(attention_mask == 0, -1e9)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = torch.sum(lstm_out * weight, dim=1)
        logits = self.cls(pooled)
        return logits

# 5. Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

# 6. 训练与评估
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        pred = torch.argmax(logits, dim=1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    return total_loss / len(loader.dataset), compute_metrics(y_true, y_pred)

@torch.no_grad()
def evaluate(model, loader, loss_fn, threshold=0.5, return_probs=False):
    model.eval()
    total_loss = 0
    y_true, y_pred, neg_probs = [], [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * input_ids.size(0)

        pred, prob = predict_with_threshold(logits, threshold=threshold)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        neg_probs.extend(prob.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred)
    if return_probs:
        return total_loss / len(loader.dataset), metrics, y_true, y_pred, neg_probs
    return total_loss / len(loader.dataset), metrics, y_true, y_pred

# 7. 维度匹配与全量预测
def match_aspects(text, aspect_keywords):
    hit = []
    for aspect, keywords in aspect_keywords.items():
        if any(k in text for k in keywords):
            hit.append(aspect)
    return hit

@torch.no_grad()
def predict_texts(model, tokenizer, texts, threshold=0.5, batch_size=64):
    model.eval()
    preds, probs = [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="全量预测", leave=False):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        pred, prob = predict_with_threshold(logits, threshold=threshold)

        preds.extend(pred.detach().cpu().numpy().tolist())
        probs.extend(prob.detach().cpu().numpy().tolist())

    return preds, probs

def build_aspect_table(df_pred, aspect_keywords):
    """
    说明：
    - 每条评论可命中多个维度
    - 每个维度分别统计该维度下的好评/差评数
    """
    records = []
    for _, row in df_pred.iterrows():
        text = row[TEXT_COL]
        pred = row["pred_label"]
        aspects = match_aspects(text, aspect_keywords)
        for asp in aspects:
            records.append({
                "维度": asp,
                "情感": "差评" if pred == 0 else "好评"
            })

    if not records:
        return pd.DataFrame(columns=["维度", "好评数", "差评数", "总数", "好评率"])

    tmp = pd.DataFrame(records)
    stat = tmp.groupby(["维度", "情感"]).size().unstack(fill_value=0)

    for col in ["好评", "差评"]:
        if col not in stat.columns:
            stat[col] = 0

    stat = stat[["好评", "差评"]].reset_index()
    stat["总数"] = stat["好评"] + stat["差评"]
    stat["好评率"] = (stat["好评"] / stat["总数"]).round(4)

    stat = stat.rename(columns={"好评": "好评数", "差评": "差评数"})
    stat = stat.sort_values("维度").reset_index(drop=True)
    return stat

def build_aspect_word_stat(df_pred, aspect_name, aspect_keywords, topn=15):
    texts = df_pred[
        df_pred[TEXT_COL].apply(lambda x: aspect_name in match_aspects(x, aspect_keywords))
    ][TEXT_COL].tolist()

    words = []
    for text in texts:
        for kw in aspect_keywords[aspect_name]:
            cnt = text.count(kw)
            if cnt > 0:
                words.extend([kw] * cnt)

    if len(words) == 0:
        return pd.Series(dtype=int)

    return pd.Series(words).value_counts().head(topn)

def packed_bubble_positions(radii, padding=0.08, max_iter=1200):
    placed = []
    if len(radii) == 0:
        return placed

    placed.append((0.0, 0.0, radii[0]))
    golden_angle = np.pi * (3 - np.sqrt(5))

    for i in range(1, len(radii)):
        r = radii[i]
        found = False

        for t in range(1, max_iter + 1):
            angle = t * golden_angle
            dist = 0.15 * np.sqrt(t)
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)

            ok = True
            for px, py, pr in placed:
                min_dist = r + pr + padding
                if (x - px) ** 2 + (y - py) ** 2 < min_dist ** 2:
                    ok = False
                    break

            if ok:
                placed.append((x, y, r))
                found = True
                break

        if not found:
            angle = i * golden_angle
            dist = 0.5 + 0.25 * i
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            placed.append((x, y, r))

    return placed

def plot_aspect_wordfreq_bubble(df_pred, aspect_name, aspect_keywords, save_path, topn=15):
    word_stat = build_aspect_word_stat(df_pred, aspect_name, aspect_keywords, topn=topn)

    labels = word_stat.index.tolist()
    values = word_stat.values.astype(float)

    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = values[order]

    radii = np.sqrt(values)
    radii = radii / radii.max() * 2.2 + 0.4
    circles = packed_bubble_positions(radii, padding=0.10, max_iter=2000)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    def get_color_by_radius(r):
        """根据气泡半径返回对应颜色"""
        if r >= 1.45:
            face = "#24B6B6"
            edge = "#1A8A8A"
            text = "#FFFFFF"
        elif r >= 1.15:
            face = "#6BCACA"
            edge = "#4DA8A8"
            text = "#FFFFFF"
        elif r >= 0.9:
            face = "#A8E0E0"
            edge = "#85CCCD"
            text = "#1A6B6B"
        else:
            face = "#E8F5F5"
            edge = "#B8D8D8"
            text = "#2E8B8B"
        return face, edge, text

    for (x, y, r), word, freq in zip(circles, labels, values):
        face_color, edge_color, text_color = get_color_by_radius(r)

        # 光晕效果
        halo = plt.Circle((x, y), r * 1.12, color=edge_color, alpha=0.08, lw=0)
        ax.add_patch(halo)

        # 气泡本体
        bubble = plt.Circle(
            (x, y), r,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(bubble)

        if r >= 1.45:
            fs = 24
        elif r >= 1.15:
            fs = 19
        elif r >= 0.9:
            fs = 16
        else:
            fs = 13

        show_word = word if len(word) <= 6 else word[:6] + "…"

        ax.text(
            x, y, show_word,
            ha="center", va="center",
            fontsize=fs, color=text_color, weight="bold",
            fontproperties = CN_FONT
        )

    xmin = min(x - r for (x, y, r) in circles) - 0.4
    xmax = max(x + r for (x, y, r) in circles) + 0.4
    ymin = min(y - r for (x, y, r) in circles) - 0.4
    ymax = max(y + r for (x, y, r) in circles) + 0.4

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] 已保存{aspect_name}词频气泡图: {save_path}")

def plot_negative_acc_curve(history, save_path):
    epochs = [x["epoch"] for x in history]
    train_neg_acc = [x["train_neg_acc"] for x in history]
    val_neg_acc = [x["val_neg_acc"] for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_neg_acc, marker="o", label="Train Negative Accuracy",color="#85CCCD")
    plt.plot(epochs, val_neg_acc, marker="o", label="Val Negative Accuracy",color="#87B814")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] 已保存差评Accuracy变化图: {save_path}")

def plot_aspect_sentiment_bar(aspect_stat, save_path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(aspect_stat))
    width = 0.35

    plt.bar(x - width / 2, aspect_stat["好评数"], width=width, label="好评数",color="#85CCCD")
    plt.bar(x + width / 2, aspect_stat["差评数"], width=width, label="差评数",color="#24B6B6")

    plt.xticks(x, aspect_stat["维度"], fontproperties=CN_FONT,fontsize=14)
    plt.yticks(fontsize=12)
    plt.ylabel("数量",fontproperties=CN_FONT,fontsize=14)
    plt.legend(prop=CN_FONT,fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 已保存四维度好评差评柱状图: {save_path}")

# 8. 主函数
def main():
    set_seed(SEED)
    print("=" * 80)
    print("DEVICE:", DEVICE)
    print("=" * 80)

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    df = load_csv_auto(DATA_PATH)
    df = prepare_dataframe(df)

    train_df, val_df, test_df = grouped_split(df, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        force_download=True
    )

    train_ds = ReviewDataset(train_df[TEXT_COL], train_df[LABEL_COL], tokenizer, MAX_LEN)
    val_ds = ReviewDataset(val_df[TEXT_COL], val_df[LABEL_COL], tokenizer, MAX_LEN)
    test_ds = ReviewDataset(test_df[TEXT_COL], test_df[LABEL_COL], tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=build_weighted_sampler(train_df[LABEL_COL].tolist()),
        num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    model = RobertaBiLSTMAttention(
        model_name=MODEL_NAME,
        num_labels=2,
        lstm_hidden=LSTM_HIDDEN,
        dropout=DROPOUT
    ).to(DEVICE)

    # 冻结底层，减轻过拟合
    freeze_encoder(model.encoder, freeze_embeddings=True, freeze_layers=6)

    # 类权重
    y_train = train_df[LABEL_COL].to_numpy()
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    loss_fn = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)

    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
        ]
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps
    )

    best_metric = -1
    best_epoch = 0
    best_threshold = 0.5
    patience_count = 0
    history = []

    # 训练
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 80}\nEpoch {epoch}/{EPOCHS}\n{'=' * 80}")

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn)

        val_loss, _, val_y, _, val_neg_probs = evaluate(
            model, val_loader, loss_fn, threshold=0.5, return_probs=True
        )
        cur_threshold, val_metrics = search_best_threshold(val_y, val_neg_probs)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_neg_acc": train_metrics["minority_acc"],
            "val_neg_acc": val_metrics["minority_acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_minority_f1": val_metrics["minority_f1"],
            "best_threshold": cur_threshold
        })

        print(f"[Train] loss={train_loss:.4f} acc={train_metrics['acc']:.4f} "
              f"macro_f1={train_metrics['macro_f1']:.4f} neg_acc={train_metrics['minority_acc']:.4f} "
              f"neg_f1={train_metrics['minority_f1']:.4f}")

        print(f"[Val]   loss={val_loss:.4f} acc={val_metrics['acc']:.4f} "
              f"macro_f1={val_metrics['macro_f1']:.4f} neg_acc={val_metrics['minority_acc']:.4f} "
              f"neg_f1={val_metrics['minority_f1']:.4f} threshold={cur_threshold:.2f}")

        # 以差评F1作为最佳模型标准
        if val_metrics["minority_f1"] > best_metric:
            best_metric = val_metrics["minority_f1"]
            best_epoch = epoch
            best_threshold = cur_threshold
            patience_count = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "threshold": best_threshold
            }, BEST_MODEL_PATH)
            print(f"[INFO] 已保存最佳模型: {BEST_MODEL_PATH}")
        else:
            patience_count += 1
            print(f"[INFO] 差评F1未提升 patience={patience_count}/{PATIENCE}")

        if patience_count >= PATIENCE:
            print("[INFO] 提前停止")
            break

    # 画图
    history_df = pd.DataFrame(history)
    history_df.to_excel("training_history.xlsx", index=False)
    plot_negative_acc_curve(history, NEG_ACC_FIG_PATH)

    print(f"\n[INFO] 最佳epoch={best_epoch}, 最佳差评F1={best_metric:.4f}, 最佳阈值={best_threshold:.2f}")

    # 测试
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = ckpt["threshold"]

    test_loss, test_metrics, y_true, y_pred = evaluate(
        model, test_loader, loss_fn, threshold=best_threshold, return_probs=False
    )

    print(f"\n{'=' * 80}\n测试集结果\n{'=' * 80}")
    print(f"Test loss      : {test_loss:.4f}")
    print(f"Test acc       : {test_metrics['acc']:.4f}")
    print(f"Test macro_f1  : {test_metrics['macro_f1']:.4f}")
    print(f"Test neg_acc   : {test_metrics['minority_acc']:.4f}")
    print(f"Test neg_f1    : {test_metrics['minority_f1']:.4f}")
    print(f"Best threshold : {best_threshold:.2f}")
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("混淆矩阵：")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    # 全量预测
    full_df = df.copy()
    full_preds, full_probs = predict_texts(
        model=model,
        tokenizer=tokenizer,
        texts=full_df[TEXT_COL].tolist(),
        threshold=best_threshold,
        batch_size=EVAL_BATCH_SIZE
    )
    full_df["pred_label"] = full_preds
    full_df["pred_prob_neg"] = full_probs
    full_df["pred_sentiment"] = full_df["pred_label"].map({0: "差评", 1: "好评"})
    full_df["命中维度"] = full_df[TEXT_COL].apply(lambda x: "、".join(match_aspects(x, ASPECT_KEYWORDS)))

    # 四维度统计
    aspect_stat = build_aspect_table(full_df, ASPECT_KEYWORDS)
    # 生成图表
    plot_aspect_sentiment_bar(aspect_stat, ASPECT_BAR_FIG_PATH)
    plot_aspect_wordfreq_bubble(full_df, "专业性", ASPECT_KEYWORDS, PROF_WORDCLOUD_FIG_PATH)
    plot_aspect_wordfreq_bubble(full_df, "安全性", ASPECT_KEYWORDS, SAFE_WORDCLOUD_FIG_PATH)

    # 导出
    with pd.ExcelWriter(ASPECT_RESULT_PATH, engine="openpyxl") as writer:
        aspect_stat.to_excel(writer, sheet_name="四维度统计", index=False)
        full_df.to_excel(writer, sheet_name="全量预测结果", index=False)
        history_df.to_excel(writer, sheet_name="训练历史", index=False)

    print(f"\n[INFO] 四维度统计表已保存: {ASPECT_RESULT_PATH}")
    print("\n四维度统计结果：")
    print(aspect_stat)

if __name__ == "__main__":
    main()