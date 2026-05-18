import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# =========================
# 0. 设定随机种子(保证可复现)
# =========================
SEED = 42
np.random.seed(SEED)

# =========================
# 1. 加载已带维度匹配的评论数据
# =========================
df = pd.read_csv('comments_with_dimensions.csv', encoding='utf-8-sig')
print(f'原始评论数: {len(df)}')

# 评论字段名(保持和原文件一致)
TEXT_COL = '评论'
LABEL_COL = 'label'

# =========================
# 2. 数据清洗:过滤极端长度评论
# =========================
df['评论长度'] = df[TEXT_COL].astype(str).apply(len)
df = df[(df['评论长度'] >= 10) & (df['评论长度'] <= 300)].copy()
print(f'长度过滤后: {len(df)} 条 (保留 10-300 字)')

# 原始好评/差评分布
print(f'\n=== 过滤后类别分布 ===')
print(df[LABEL_COL].value_counts())
print(f'好评比例: {(df[LABEL_COL] == 1).sum() / len(df) * 100:.2f}%')
print(f'差评比例: {(df[LABEL_COL] == 0).sum() / len(df) * 100:.2f}%')

# =========================
# 3. 分层抽样
# 目标: 3500 条,按原比例分层(好评 91.98% / 差评 8.02%)
# =========================
TOTAL_SAMPLES = 3500

# 按原比例计算
pos_ratio = (df[LABEL_COL] == 1).sum() / len(df)
neg_ratio = (df[LABEL_COL] == 0).sum() / len(df)
n_pos = round(TOTAL_SAMPLES * pos_ratio)
n_neg = TOTAL_SAMPLES - n_pos

print(f'\n=== 分层抽样目标 ===')
print(f'  好评: {n_pos} 条 ({pos_ratio * 100:.2f}%)')
print(f'  差评: {n_neg} 条 ({neg_ratio * 100:.2f}%)')

# 检查差评是否足够
available_neg = (df[LABEL_COL] == 0).sum()
if n_neg > available_neg:
    print(f'  警告: 差评不足,将全部 {available_neg} 条差评纳入抽样')
    n_neg = available_neg
    n_pos = TOTAL_SAMPLES - n_neg

# 分层抽样
pos_df = df[df[LABEL_COL] == 1].sample(n=n_pos, random_state=SEED)
neg_df = df[df[LABEL_COL] == 0].sample(n=n_neg, random_state=SEED)
sampled_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f'\n=== 实际抽样结果 ===')
print(f'总数: {len(sampled_df)}')
print(f'  好评: {(sampled_df[LABEL_COL] == 1).sum()}')
print(f'  差评: {(sampled_df[LABEL_COL] == 0).sum()}')

# =========================
# 4. 抽样质量检查:各维度覆盖
# =========================
print(f'\n=== 抽样后各维度覆盖 ===')
for dim in ['专业性', '安全性', '响应性', '服务性']:
    col = f'命中_{dim}'
    if col in sampled_df.columns:
        n = sampled_df[col].sum()
        print(f'  {dim}: {n} 条 ({n / len(sampled_df) * 100:.2f}%)')

zero_match = (sampled_df['命中维度数'] == 0).sum() if '命中维度数' in sampled_df.columns else 0
print(f'\n  0 个维度命中: {zero_match} 条 ({zero_match / len(sampled_df) * 100:.2f}%)')

# =========================
# 5. 三人任务分配
#
# 策略:
# - 共标 500 条:三人都标,用于算 Fleiss Kappa
# - 独标 3000 条:平均分配给三人,每人 1000 条
# - 每人总工作量 = 共标 500 + 独标 1000 = 1500 条
# =========================
SHARED_N = 500  # 共标数量(算 Kappa 用)
INDIVIDUAL_PER_PERSON = 1000  # 每人独标数量
ANNOTATORS = ['A', 'B', 'C']

# 重新打乱后切分
shuffled = sampled_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# 前 500 条作为共标
shared_df = shuffled.iloc[:SHARED_N].copy()
# 后面的 3000 条平均分给 3 个人
individual_dfs = {}
start = SHARED_N
for i, annotator in enumerate(ANNOTATORS):
    end = start + INDIVIDUAL_PER_PERSON
    individual_dfs[annotator] = shuffled.iloc[start:end].copy()
    start = end

print(f'\n=== 任务分配 ===')
print(f'共标(三人都标): {len(shared_df)} 条')
for annotator in ANNOTATORS:
    print(f'  标注员 {annotator} 独标: {len(individual_dfs[annotator])} 条')


# =========================
# 6. 输出标注任务 csv
#
# 每个标注员收到的文件包含:
# - 共标 500 条 + 自己的独标 1000 条 = 1500 条
# - 顺序打乱,标注员看不出哪些是共标
# - 新增"情感标注"空列,标注员填 1(好评) 或 0(差评)
# =========================
def prepare_annotation_file(annotator):
    own_individual = individual_dfs[annotator].copy()
    own_individual['任务类型'] = '独标'

    shared_copy = shared_df.copy()
    shared_copy['任务类型'] = '共标'

    combined = pd.concat([own_individual, shared_copy], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED + ord(annotator)).reset_index(drop=True)

    # 准备标注模板:只保留必要列,加上空的"情感标注"列
    output = pd.DataFrame({
        '序号': range(1, len(combined) + 1),
        '评论': combined[TEXT_COL],
        '原始评分(参考)': combined[LABEL_COL].map({1: '5星', 0: '1-4星'}),
        '情感标注': '',  # 标注员填: 1=好评, 0=差评
        '任务类型': combined['任务类型'],  # 标注完后用于区分独标和共标
        '_internal_id': combined.index  # 内部 ID,用于后续合并(标注员不用管)
    })

    return output, combined


for annotator in ANNOTATORS:
    annotation_file, _ = prepare_annotation_file(annotator)
    filename = f'sentiment_annotation_标注员{annotator}.csv'
    annotation_file.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f'已保存: {filename}')

# =========================
# 7. 输出共标对照表(供后续 Kappa 计算)
# =========================
shared_reference = shared_df[[TEXT_COL, LABEL_COL]].copy()
shared_reference.columns = ['评论', '原始评分(参考)']
shared_reference['原始评分(参考)'] = shared_reference['原始评分(参考)'].map({1: '5星', 0: '1-4星'})
shared_reference.insert(0, '共标序号', range(1, len(shared_reference) + 1))
shared_reference.to_csv('shared_reference_for_kappa.csv', index=False, encoding='utf-8-sig')
print(f'已保存: shared_reference_for_kappa.csv (共标 500 条参考,用于 Kappa 计算)')

# =========================
# 8. 完整抽样记录(供论文展示)
# =========================
sampled_df['任务分配'] = ''
sampled_df.loc[sampled_df.index.isin(shared_df.index), '任务分配'] = '共标'
for annotator in ANNOTATORS:
    sampled_df.loc[sampled_df.index.isin(individual_dfs[annotator].index), '任务分配'] = f'独标-{annotator}'

sampled_df.to_csv('sampling_record.csv', index=False, encoding='utf-8-sig')
print(f'已保存: sampling_record.csv (完整抽样记录)')

# =========================
# 9. 抽样统计摘要(论文写作用)
# =========================
print(f'\n{"=" * 50}')
print(f'抽样统计摘要(可写入论文)')
print(f'{"=" * 50}')
print(f'原始评论总数: {len(df) + (df["评论长度"].isnull().sum() if "评论长度" in df.columns else 0)}')
print(f'有效评论数(长度 10-300 字): {len(df)}')
print(f'')
print(f'抽样总数: {len(sampled_df)}')
print(
    f'好评: {(sampled_df[LABEL_COL] == 1).sum()} 条 ({(sampled_df[LABEL_COL] == 1).sum() / len(sampled_df) * 100:.2f}%)')
print(
    f'差评: {(sampled_df[LABEL_COL] == 0).sum()} 条 ({(sampled_df[LABEL_COL] == 0).sum() / len(sampled_df) * 100:.2f}%)')
print(f'')
print(f'分配方案:')
print(f'  共标(三人重叠): {len(shared_df)} 条')
for annotator in ANNOTATORS:
    print(f'  标注员 {annotator} 独标: {len(individual_dfs[annotator])} 条')
print(f'  每人实际标注: {SHARED_N + INDIVIDUAL_PER_PERSON} 条')
print(f'')
print(f'维度覆盖:')
for dim in ['专业性', '安全性', '响应性', '服务性']:
    col = f'命中_{dim}'
    if col in sampled_df.columns:
        n = sampled_df[col].sum()
        print(f'  {dim}: {n} 条 ({n / len(sampled_df) * 100:.2f}%)')