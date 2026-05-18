import pandas as pd
import re
from gensim.models import Word2Vec
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

# =========================
# 1. 加载已训练好的 Word2Vec 模型(不重训练)
# =========================
model = Word2Vec.load('word2vec_medical_o2o.model')
print(f'已加载词向量模型,词汇表大小: {len(model.wv.key_to_index)}')

# =========================
# 2. 读取种子词
# =========================
seed_df = pd.read_excel('kappa_annotation_result.xlsx', sheet_name='已确定种子词库')
seed_df = seed_df.dropna(subset=['关键词', '最终维度']).copy()
valid_dims = ['专业性', '安全性', '响应性', '服务性']
seed_df = seed_df[seed_df['最终维度'].isin(valid_dims)].copy()
seed_df['关键词'] = seed_df['关键词'].astype(str).str.strip()
seed_df['最终维度'] = seed_df['最终维度'].astype(str).str.strip()

print(f'种子词总数: {len(seed_df)}')

# =========================
# 3. 调严参数 + 添加黑名单过滤
# =========================
TOP_N = 8  # 从 15 减到 8
SIMILARITY_THRESHOLD = 0.65  # 从 0.5 提高到 0.65


# 黑名单:这些词不能进入扩充词库
def is_invalid_word(word):
    """判断是否是无效词(碎片、数字、时间、地名等)"""
    # 1. 长度过滤:1 字或超过 6 字
    if len(word) < 2 or len(word) > 6:
        return True
    # 2. 含字母或数字的混合词
    if re.search(r'[a-zA-Z0-9]', word):
        return True
    # 3. 时间词/量词
    time_patterns = [
        r'^[一二三四五六七八九十百千万0-9]+',  # 以数字开头
        r'[点分秒时天年月周日早晚]$',  # 时间结尾
        r'[次个根针条款种位张]$',  # 量词结尾
    ]
    for p in time_patterns:
        if re.search(p, word):
            return True
    # 4. 单字 + 通用词的组合(如"一会""不太""不算")
    common_prefixes = ['一', '不', '两', '三', '几', '多', '太', '很', '挺', '蛮']
    if word[0] in common_prefixes and len(word) <= 3:
        return True
    # 5. 通用名词黑名单
    blacklist = {
        '产品', '商品', '中国', '上海', '北京', '业务', '公司', '平台',
        '互联网', '社会', '事情', '事项', '内容', '方面', '问题',
        '人员', '人士', '人群', '家人', '老人', '小孩', '宝宝',
        '一般', '一会', '一切', '一直', '一遍', '一定', '所有',
        '今天', '昨天', '明天', '现在', '以后', '之前', '之后',
        '这里', '那里', '什么', '怎么', '为何', '应该', '可能'
    }
    if word in blacklist:
        return True
    return False


# =========================
# 4. 扩充关键词库
# =========================
expanded_keywords = {dim: set() for dim in valid_dims}
for _, row in seed_df.iterrows():
    expanded_keywords[row['最终维度']].add(row['关键词'])

expansion_log = []
skipped_invalid = 0

for _, row in seed_df.iterrows():
    seed_word = row['关键词']
    seed_dim = row['最终维度']

    if seed_word not in model.wv:
        continue

    similar_words = model.wv.most_similar(seed_word, topn=TOP_N)

    for sim_word, sim_score in similar_words:
        if sim_score < SIMILARITY_THRESHOLD:
            continue
        # 关键:加入黑名单过滤
        if is_invalid_word(sim_word):
            skipped_invalid += 1
            continue
        if sim_word not in expanded_keywords[seed_dim]:
            expanded_keywords[seed_dim].add(sim_word)
            expansion_log.append({
                '扩充词': sim_word,
                '来源种子词': seed_word,
                '相似度': round(sim_score, 4),
                '所属维度': seed_dim
            })

# 跨维度去重
expansion_df = pd.DataFrame(expansion_log)
if len(expansion_df) > 0:
    expansion_df = expansion_df.sort_values('相似度', ascending=False)
    expansion_df_dedup = expansion_df.drop_duplicates(subset=['扩充词'], keep='first')

    final_keywords = {dim: set() for dim in valid_dims}
    for _, row in seed_df.iterrows():
        final_keywords[row['最终维度']].add(row['关键词'])
    for _, row in expansion_df_dedup.iterrows():
        final_keywords[row['所属维度']].add(row['扩充词'])
else:
    final_keywords = expanded_keywords
    expansion_df_dedup = pd.DataFrame()

# =========================
# 5. 输出统计
# =========================
print(f'\n=== 扩充结果(参数调严 + 黑名单过滤) ===')
print(f'黑名单过滤掉的噪声词: {skipped_invalid} 个')
print(f'扩充后关键词总数: {sum(len(v) for v in final_keywords.values())}')

for dim in valid_dims:
    seed_count = (seed_df['最终维度'] == dim).sum()
    total_count = len(final_keywords[dim])
    expanded_count = total_count - seed_count
    print(f'  {dim}: 种子词 {seed_count} + 扩充词 {expanded_count} = 总计 {total_count}')

# =========================
# 6. 导出 csv(供后续维度匹配用)
# =========================
keyword_records = []
for dim, words in final_keywords.items():
    for word in sorted(words):
        is_seed = word in seed_df['关键词'].tolist()
        keyword_records.append({
            '维度': dim,
            '关键词': word,
            '类型': '种子词' if is_seed else '扩充词'
        })

result_df = pd.DataFrame(keyword_records)
result_df.to_csv('keyword_library_final_v2.csv', index=False, encoding='utf-8-sig')
print(f'\n已保存: keyword_library_final_v2.csv')

# 同时导出扩充明细(供论文附录)
if len(expansion_df_dedup) > 0:
    expansion_df_dedup.to_csv('expansion_detail_v2.csv', index=False, encoding='utf-8-sig')
    print(f'已保存: expansion_detail_v2.csv')