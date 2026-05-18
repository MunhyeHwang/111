# -*- coding: utf-8 -*-
"""
RoBERTa-BiLSTM-Attention 中文评论情感分析(路 B 修订版)
功能：
1) 使用人工标注的 2909 条数据训练模型(8:1:1 划分)
2) 在测试集上评估性能(F1/Recall/混淆矩阵)
3) 对剩余未标注的 4923 条评论做情感预测(不重新预测训练数据)
4) 合并 2909 人工 + 4923 模型预测 = 7832 全量带标签数据
5) 按四个维度统计两个版本的情感分布:
   - 仅人工标注版本(2909 条)
   - 全量版本(7832 条)
"""

import os
import re
import random
import warnings
from pathlib import Path
import gc

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
import matplotlib
from matplotlib.font_manager import FontProperties

# =========================
# 字体设置(保留你原代码)
# =========================
matplotlib.font_manager._load_fontmanager(try_read_cache=False)
font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf")
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()
matplotlib.rcParams['font.family'] = font_name
matplotlib.rcParams['axes.unicode_minus'] = False
CN_FONT = FontProperties(fname=font_path)
print(f"[INFO] 当前字体: {font_name}")

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# =========================
# 1. 配置
# =========================
SEED = 42
MODEL_NAME = "./chinese-roberta-wwm-ext"

# 数据路径(关键改动)
ANNOTATED_DATA_PATH = "sentiment_annotated_final.csv"  # 2909 条人工标注
FULL_DATA_PATH = "huizong_cleaned.csv"  # 7832 条全量评论
KEYWORDS_PATH = "keyword_library_final_v3.csv"  # 256 个维度关键词
TEXT_COL = "评论"
LABEL_COL = "最终情感"  # 修改:从 "lable" 改为 "最终情感"

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
MINORITY_CLASS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 使用设备: {DEVICE}")

BEST_MODEL_PATH = "best_roberta_bilstm_attn.pt"
TEST_REPORT_PATH = "测试集评估报告.xlsx"
NEG_RECALL_FIG_PATH = "差评召回率变化曲线.png"
ASPECT_RESULT_HUMAN_PATH = "四维度情感统计_仅人工标注.xlsx"
ASPECT_RESULT_FULL_PATH = "四维度情感统计_全量数据.xlsx"
ABLATION_RESULT_PATH = "消融实验结果.xlsx"
ASPECT_BAR_FIG_PATH = "四维度情感分布柱状图.png"
FULL_PREDICTIONS_PATH = "全量数据情感标签.csv"

# 设置随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# 2. 加载维度关键词(从 v3 文件加载,而不是硬编码)
# =========================
keyword_df = pd.read_csv(KEYWORDS_PATH, encoding='utf-8-sig')
ASPECT_KEYWORDS = {}
for dim in ['专业性', '安全性', '响应性', '服务性']:
    words = keyword_df[keyword_df['维度'] == dim]['关键词'].tolist()
    ASPECT_KEYWORDS[dim] = words
    print(f"[INFO] {dim}: {len(words)} 个关键词")


# =========================
# 3. 数据加载与清洗
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9，。！？、：；""''（）\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 加载人工标注数据(2909 条)
print(f"\n[INFO] 加载人工标注数据: {ANNOTATED_DATA_PATH}")
df_labeled = pd.read_csv(ANNOTATED_DATA_PATH, encoding='utf-8-sig')
df_labeled[TEXT_COL] = df_labeled[TEXT_COL].astype(str).apply(clean_text)
df_labeled = df_labeled[df_labeled[TEXT_COL].str.len() >= 5].copy()
df_labeled[LABEL_COL] = df_labeled[LABEL_COL].astype(int)
print(f"[INFO] 人工标注数据条数: {len(df_labeled)}")
print(f"[INFO] 好评比例: {(df_labeled[LABEL_COL] == 1).mean() * 100:.2f}%")
print(f"[INFO] 差评比例: {(df_labeled[LABEL_COL] == 0).mean() * 100:.2f}%")

# 加载全量数据(7832 条,用于后续预测未标注部分)
print(f"\n[INFO] 加载全量数据: {FULL_DATA_PATH}")
df_full = pd.read_csv(FULL_DATA_PATH, encoding='gbk').rename(columns={'lable': 'label'})
df_full = df_full[['评论']].dropna().copy()
df_full['评论'] = df_full['评论'].astype(str).apply(clean_text)
df_full = df_full[df_full['评论'].str.len() >= 5].copy()
print(f"[INFO] 全量数据条数: {len(df_full)}")

# =========================
# 4. 数据集划分(8:1:1)
# =========================
print(f"\n[INFO] 数据集划分(8:1:1)...")
train_df, temp_df = train_test_split(
    df_labeled,
    test_size=0.2,
    stratify=df_labeled[LABEL_COL],
    random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df[LABEL_COL],
    random_state=SEED
)
print(
    f"  训练集: {len(train_df)} 条 (好评 {(train_df[LABEL_COL] == 1).sum()} / 差评 {(train_df[LABEL_COL] == 0).sum()})")
print(f"  验证集: {len(val_df)} 条 (好评 {(val_df[LABEL_COL] == 1).sum()} / 差评 {(val_df[LABEL_COL] == 0).sum()})")
print(f"  测试集: {len(test_df)} 条 (好评 {(test_df[LABEL_COL] == 1).sum()} / 差评 {(test_df[LABEL_COL] == 0).sum()})")

# 找出未标注部分(全量 - 人工标注)
annotated_texts = set(df_labeled[TEXT_COL].tolist())
unlabeled_df = df_full[~df_full['评论'].isin(annotated_texts)].copy().reset_index(drop=True)
print(f"  未标注集(待预测): {len(unlabeled_df)} 条")

# =========================
# 5. Tokenizer 加载
# =========================
print(f"\n[INFO] 加载 tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================
# 6. Dataset 类
# =========================
class CommentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =========================
# 7. WeightedRandomSampler 构造
# =========================
def build_weighted_sampler(labels):
    class_count = np.bincount(labels)
    class_weights = 1.0 / class_count
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# =========================
# 8. 模型定义
# =========================
class RoBERTaBiLSTMAttention(nn.Module):
    def __init__(self, model_name, hidden_dim=LSTM_HIDDEN, num_classes=2,
                 dropout=DROPOUT, use_bilstm=True, use_attention=True):
        super().__init__()
        self.use_bilstm = use_bilstm
        self.use_attention = use_attention
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        if use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=encoder_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            seq_dim = hidden_dim * 2
        else:
            seq_dim = encoder_dim

        if use_attention:
            self.attn_proj = nn.Linear(seq_dim, seq_dim)
            self.attn_score = nn.Linear(seq_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(seq_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state

        if self.use_bilstm:
            seq_out, _ = self.bilstm(seq_out)

        if self.use_attention:
            mask = attention_mask.unsqueeze(-1).bool()
            score = torch.tanh(self.attn_proj(seq_out))
            score = self.attn_score(score)
            score = score.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(score, dim=1)
            pooled = (seq_out * attn_weights).sum(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (seq_out * mask).sum(dim=1) / mask.sum(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# =========================
# 9. Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


# =========================
# 10. 训练和评估函数
# =========================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    macro_f1 = f1.mean()
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'neg_precision': precision[0],
        'neg_recall': recall[0],
        'neg_f1': f1[0],
        'pos_precision': precision[1],
        'pos_recall': recall[1],
        'pos_f1': f1[1]
    }


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=-1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
    return total_loss / len(loader.dataset), compute_metrics(y_true, y_pred)


def evaluate(model, loader, loss_fn, threshold=0.5, return_probs=False):
    model.eval()
    total_loss = 0
    y_true, y_pred, neg_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            probs = torch.softmax(logits, dim=-1)
            neg_prob = probs[:, 0]
            preds = (neg_prob >= threshold).long()  # 自定义阈值
            preds = 1 - preds  # 转回 0/1: 差评=0, 好评=1
            # 上面 2 行: neg_prob >= threshold 时预测为差评(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            neg_probs.extend(neg_prob.cpu().numpy().tolist())
    metrics = compute_metrics(y_true, y_pred)
    if return_probs:
        return total_loss / len(loader.dataset), metrics, y_true, y_pred, neg_probs
    return total_loss / len(loader.dataset), metrics, y_true, y_pred


# =========================
# 11. 预测函数(用于未标注数据)
# =========================
def predict_unlabeled(model, texts, threshold=0.5):
    """对未标注文本预测情感标签"""
    model.eval()
    dataset = CommentDataset(texts.tolist(), labels=None, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting unlabeled'):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            neg_prob = probs[:, 0]
            preds = (neg_prob < threshold).long()  # neg_prob < threshold 时为好评(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(neg_prob.cpu().numpy().tolist())
    return all_preds, all_probs


# =========================
# 12. 训练流程封装
# =========================
def train_model(use_bilstm=True, use_attention=True, model_tag='Full'):
    """训练一个模型变体并返回测试集指标"""
    print(f"\n{'=' * 60}")
    print(f"训练模型: {model_tag} (BiLSTM={use_bilstm}, Attention={use_attention})")
    print(f"{'=' * 60}")

    # 重置随机种子保证可复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 构造 dataset 和 dataloader
    train_dataset = CommentDataset(
        train_df[TEXT_COL].tolist(),
        train_df[LABEL_COL].tolist(),
        tokenizer
    )
    val_dataset = CommentDataset(
        val_df[TEXT_COL].tolist(),
        val_df[LABEL_COL].tolist(),
        tokenizer
    )
    test_dataset = CommentDataset(
        test_df[TEXT_COL].tolist(),
        test_df[LABEL_COL].tolist(),
        tokenizer
    )

    sampler = build_weighted_sampler(np.array(train_df[LABEL_COL].tolist()))
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # 模型
    model = RoBERTaBiLSTMAttention(
        MODEL_NAME, use_bilstm=use_bilstm, use_attention=use_attention
    ).to(DEVICE)

    # 类别权重 + Focal Loss
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array([0, 1]),
        y=np.array(train_df[LABEL_COL].tolist())
    )
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    loss_fn = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)

    # 分层学习率
    encoder_params = list(model.encoder.named_parameters())
    head_params = [(n, p) for n, p in model.named_parameters() if not n.startswith('encoder.')]
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY, 'lr': ENCODER_LR},
        {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': ENCODER_LR},
        {'params': [p for n, p in head_params if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY, 'lr': HEAD_LR},
        {'params': [p for n, p in head_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': HEAD_LR}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # 训练循环 + 早停
    best_val_f1 = -1
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_neg_recall': []}

    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn
        )
        val_loss, val_metrics, _, _, val_neg_probs = evaluate(
            model, val_loader, loss_fn, threshold=0.5, return_probs=True
        )

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MacroF1: {val_metrics['macro_f1']:.4f} | "
              f"Val NegRecall: {val_metrics['neg_recall']:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_neg_recall'].append(val_metrics['neg_recall'])

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0
            if model_tag == 'Full':
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  → 保存最优模型到 {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  → 早停触发(连续 {PATIENCE} 轮 F1 未提升)")
                break

    # 阈值优化(验证集上找最优 threshold)
    print("\n[INFO] 阈值优化...")
    _, _, _, _, val_neg_probs = evaluate(
        model, val_loader, loss_fn, threshold=0.5, return_probs=True
    )
    val_labels = np.array(val_df[LABEL_COL].tolist())
    best_threshold = 0.5
    best_neg_recall = 0
    for thresh in np.arange(0.2, 0.81, 0.05):
        preds = (np.array(val_neg_probs) >= thresh).astype(int)
        preds = 1 - preds  # 转回 0/1
        m = compute_metrics(val_labels, preds)
        if m['neg_recall'] > best_neg_recall:
            best_neg_recall = m['neg_recall']
            best_threshold = thresh
    print(f"  最优阈值: {best_threshold:.2f}, 对应差评召回率: {best_neg_recall:.4f}")

    # 测试集评估
    print("\n[INFO] 测试集评估...")
    _, test_metrics, test_y_true, test_y_pred = evaluate(
        model, test_loader, loss_fn, threshold=best_threshold
    )
    print(f"测试集指标:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(test_y_true, test_y_pred)
    print(f"\n混淆矩阵:")
    print(f"               预测差评  预测好评")
    print(f"  实际差评     {cm[0, 0]:6d}    {cm[0, 1]:6d}")
    print(f"  实际好评     {cm[1, 0]:6d}    {cm[1, 1]:6d}")

    return model, test_metrics, best_threshold, cm, history


# =========================
# 13. 主流程:消融实验
# =========================
print("\n" + "=" * 60)
print("开始消融实验")
print("=" * 60)

ablation_results = []

# 完整模型
model_full, metrics_full, threshold_full, cm_full, history_full = train_model(
    use_bilstm=True, use_attention=True, model_tag='Full'
)
ablation_results.append({
    '模型': 'RoBERTa-BiLSTM-Attention',
    'Accuracy': metrics_full['accuracy'],
    'Macro_F1': metrics_full['macro_f1'],
    'Weighted_F1': metrics_full['weighted_f1'],
    '差评Recall': metrics_full['neg_recall'],
    '差评F1': metrics_full['neg_f1']
})

# 消融:去掉 BiLSTM
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
model_no_bilstm, metrics_no_bilstm, _, _, _ = train_model(
    use_bilstm=False, use_attention=True, model_tag='NoBiLSTM'
)
ablation_results.append({
    '模型': 'RoBERTa-Attention',
    'Accuracy': metrics_no_bilstm['accuracy'],
    'Macro_F1': metrics_no_bilstm['macro_f1'],
    'Weighted_F1': metrics_no_bilstm['weighted_f1'],
    '差评Recall': metrics_no_bilstm['neg_recall'],
    '差评F1': metrics_no_bilstm['neg_f1']
})

# 消融:去掉 Attention
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
model_no_attn, metrics_no_attn, _, _, _ = train_model(
    use_bilstm=True, use_attention=False, model_tag='NoAttention'
)
ablation_results.append({
    '模型': 'RoBERTa-BiLSTM',
    'Accuracy': metrics_no_attn['accuracy'],
    'Macro_F1': metrics_no_attn['macro_f1'],
    'Weighted_F1': metrics_no_attn['weighted_f1'],
    '差评Recall': metrics_no_attn['neg_recall'],
    '差评F1': metrics_no_attn['neg_f1']
})

# 保存消融实验结果
ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_excel(ABLATION_RESULT_PATH, index=False)
print(f"\n[INFO] 消融实验结果已保存: {ABLATION_RESULT_PATH}")
print(ablation_df)

# =========================
# 14. 加载最优模型,对未标注数据预测
# =========================
print("\n" + "=" * 60)
print("使用最优模型对未标注数据(4923 条)进行情感预测")
print("=" * 60)

model_best = RoBERTaBiLSTMAttention(MODEL_NAME, use_bilstm=True, use_attention=True).to(DEVICE)
model_best.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model_best.eval()

unlabeled_preds, unlabeled_probs = predict_unlabeled(
    model_best, unlabeled_df['评论'], threshold=threshold_full
)
unlabeled_df['情感'] = unlabeled_preds
unlabeled_df['来源'] = '模型预测'
print(f"[INFO] 未标注数据预测完成,好评: {sum(unlabeled_preds)}, 差评: {len(unlabeled_preds) - sum(unlabeled_preds)}")

# =========================
# 15. 合并:人工标注 2909 + 模型预测 4923 = 全量 7832
# =========================
human_part = df_labeled[[TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: '情感'}).copy()
human_part['来源'] = '人工标注'

full_predictions = pd.concat([human_part, unlabeled_df[['评论', '情感', '来源']]], ignore_index=True)
full_predictions.to_csv(FULL_PREDICTIONS_PATH, index=False, encoding='utf-8-sig')
print(f"\n[INFO] 全量数据情感标签已保存: {FULL_PREDICTIONS_PATH}")
print(f"  总条数: {len(full_predictions)}")
print(f"  来源分布:\n{full_predictions['来源'].value_counts()}")
print(f"  情感分布:")
print(f"    好评: {(full_predictions['情感'] == 1).sum()} 条 ({(full_predictions['情感'] == 1).mean() * 100:.2f}%)")
print(f"    差评: {(full_predictions['情感'] == 0).sum()} 条 ({(full_predictions['情感'] == 0).mean() * 100:.2f}%)")


# =========================
# 16. 维度归属(用关键词正则匹配)
# =========================
def build_dim_patterns(aspect_keywords):
    patterns = {}
    for dim, words in aspect_keywords.items():
        sorted_words = sorted(words, key=len, reverse=True)
        escaped = [re.escape(w) for w in sorted_words]
        patterns[dim] = re.compile('|'.join(escaped))
    return patterns


dim_patterns = build_dim_patterns(ASPECT_KEYWORDS)


def get_aspects(text):
    matched = []
    for dim, pattern in dim_patterns.items():
        if pattern.search(text):
            matched.append(dim)
    return matched


# 对人工标注 + 全量数据分别打标维度
human_part['维度'] = human_part['评论'].apply(get_aspects)
full_predictions['维度'] = full_predictions['评论'].apply(get_aspects)


# =========================
# 17. 按维度统计情感分布(两个版本)
# =========================
def aspect_sentiment_stats(df, version_name):
    """按维度统计好/差评分布"""
    stats = []
    for dim in ['专业性', '安全性', '响应性', '服务性']:
        sub = df[df['维度'].apply(lambda x: dim in x)]
        n_total = len(sub)
        n_pos = (sub['情感'] == 1).sum()
        n_neg = (sub['情感'] == 0).sum()
        pos_rate = n_pos / n_total * 100 if n_total > 0 else 0
        stats.append({
            '维度': dim,
            '评论总数': n_total,
            '好评数': n_pos,
            '差评数': n_neg,
            '好评率(%)': round(pos_rate, 2),
            '差评率(%)': round(100 - pos_rate, 2)
        })
    stats_df = pd.DataFrame(stats)
    print(f"\n=== {version_name} 各维度情感分布 ===")
    print(stats_df)
    return stats_df


# 仅人工标注版本(2909 条)
human_stats = aspect_sentiment_stats(human_part, "仅人工标注版本")
human_stats.to_excel(ASPECT_RESULT_HUMAN_PATH, index=False)
print(f"[INFO] 已保存: {ASPECT_RESULT_HUMAN_PATH}")

# 全量版本(7832 条)
full_stats = aspect_sentiment_stats(full_predictions, "全量数据版本")
full_stats.to_excel(ASPECT_RESULT_FULL_PATH, index=False)
print(f"[INFO] 已保存: {ASPECT_RESULT_FULL_PATH}")

# =========================
# 18. 可视化:维度情感分布柱状图(全量版本)
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(full_stats))
width = 0.35
ax.bar(x - width / 2, full_stats['好评数'], width, label='好评', color='#4472C4')
ax.bar(x + width / 2, full_stats['差评数'], width, label='差评', color='#C00000')
ax.set_xticks(x)
ax.set_xticklabels(full_stats['维度'], fontproperties=CN_FONT, fontsize=12)
ax.set_ylabel('评论数量', fontproperties=CN_FONT, fontsize=12)
ax.set_title('四维度好评/差评分布(全量数据)', fontproperties=CN_FONT, fontsize=14)
ax.legend(prop=CN_FONT)
plt.tight_layout()
plt.savefig(ASPECT_BAR_FIG_PATH, dpi=300)
plt.close()
print(f"[INFO] 柱状图已保存: {ASPECT_BAR_FIG_PATH}")

# =========================
# 19. 输出测试集详细评估报告(Excel)
# =========================
test_report_data = {
    '指标': ['Accuracy', 'Macro F1', 'Weighted F1', '差评 Precision', '差评 Recall', '差评 F1',
             '好评 Precision', '好评 Recall', '好评 F1', '最优阈值'],
    '数值': [
        round(metrics_full['accuracy'], 4),
        round(metrics_full['macro_f1'], 4),
        round(metrics_full['weighted_f1'], 4),
        round(metrics_full['neg_precision'], 4),
        round(metrics_full['neg_recall'], 4),
        round(metrics_full['neg_f1'], 4),
        round(metrics_full['pos_precision'], 4),
        round(metrics_full['pos_recall'], 4),
        round(metrics_full['pos_f1'], 4),
        round(threshold_full, 2)
    ]
}
test_report_df = pd.DataFrame(test_report_data)

cm_df = pd.DataFrame(
    cm_full,
    index=['实际差评', '实际好评'],
    columns=['预测差评', '预测好评']
)

with pd.ExcelWriter(TEST_REPORT_PATH) as writer:
    test_report_df.to_excel(writer, sheet_name='测试集指标', index=False)
    cm_df.to_excel(writer, sheet_name='混淆矩阵')

print(f"\n[INFO] 测试集评估报告已保存: {TEST_REPORT_PATH}")

print("\n" + "=" * 60)
print("✓ 全流程结束")
print("=" * 60)
print(f"\n输出文件清单:")
print(f"  1. {BEST_MODEL_PATH}  →  最优模型参数")
print(f"  2. {TEST_REPORT_PATH}  →  测试集评估报告")
print(f"  3. {ABLATION_RESULT_PATH}  →  消融实验结果")
print(f"  4. {FULL_PREDICTIONS_PATH}  →  全量数据情感标签")
print(f"  5. {ASPECT_RESULT_HUMAN_PATH}  →  仅人工标注的维度情感分布")
print(f"  6. {ASPECT_RESULT_FULL_PATH}  →  全量数据的维度情感分布")
print(f"  7. {ASPECT_BAR_FIG_PATH}  →  维度情感分布柱状图")