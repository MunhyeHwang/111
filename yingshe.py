import pandas as pd
import re

# =========================
# 1. 加载关键词库
# =========================
keyword_df = pd.read_csv('keyword_library_final_v3.csv', encoding='utf-8-sig')

# 构建维度→关键词列表的字典
dimension_keywords = {}
for dim in ['专业性', '安全性', '响应性', '服务性']:
    words = keyword_df[keyword_df['维度']==dim]['关键词'].tolist()
    dimension_keywords[dim] = words
    print(f'{dim}: {len(words)} 个关键词')

# =========================
# 2. 构建正则 pattern(每个维度一个)
# =========================
def build_pattern(words):
    # 按长度降序排,优先匹配长词(避免"专业"先于"专业技能"被匹配)
    sorted_words = sorted(words, key=len, reverse=True)
    escaped = [re.escape(w) for w in sorted_words]
    return re.compile('|'.join(escaped))

patterns = {dim: build_pattern(words) for dim, words in dimension_keywords.items()}

# =========================
# 3. 加载评论数据
# =========================
df = pd.read_csv('huizong_cleaned.csv', encoding='gbk').rename(columns={'lable': 'label'})
df = df[['评论', 'label']].dropna().copy()
df['评论'] = df['评论'].astype(str)
print(f'\n评论总数: {len(df)}')

# =========================
# 4. 维度匹配函数
# =========================
def match_dimensions(text):
    """返回评论涉及的维度列表(多标签)"""
    matched = []
    for dim, pattern in patterns.items():
        if pattern.search(text):
            matched.append(dim)
    return matched

def get_matched_words(text):
    """返回评论中实际命中的关键词(供调试和论文展示)"""
    result = {}
    for dim, pattern in patterns.items():
        words = pattern.findall(text)
        if words:
            result[dim] = list(set(words))
    return result

# =========================
# 5. 批量匹配
# =========================
print('\n正在匹配...')
df['matched_dimensions'] = df['评论'].apply(match_dimensions)
df['matched_words'] = df['评论'].apply(get_matched_words)

# 4 个维度作为独立的 0/1 列
for dim in dimension_keywords.keys():
    df[f'命中_{dim}'] = df['matched_dimensions'].apply(lambda x: 1 if dim in x else 0)

df['命中维度数'] = df['matched_dimensions'].apply(len)

# =========================
# 6. 统计结果
# =========================
print('\n=== 各维度命中分布 ===')
for dim in dimension_keywords.keys():
    n = df[f'命中_{dim}'].sum()
    print(f'  {dim}: {n} 条 ({n/len(df)*100:.2f}%)')

print(f'\n=== 评论命中维度数分布 ===')
for n in range(5):
    cnt = (df['命中维度数'] == n).sum()
    print(f'  命中 {n} 个维度: {cnt} 条 ({cnt/len(df)*100:.2f}%)')

# =========================
# 7. 保存结果
# =========================
# 把 list 和 dict 转成字符串,方便存csv
df['matched_dimensions'] = df['matched_dimensions'].apply(lambda x: ','.join(x) if x else '')
df['matched_words'] = df['matched_words'].apply(str)

df.to_csv('comments_with_dimensions.csv', index=False, encoding='utf-8-sig')
print('\n已保存: comments_with_dimensions.csv')

# =========================
# 8. 抽样检查(供论文展示)
# =========================
print('\n=== 抽样展示 5 条命中评论 ===')
sample = df[df['命中维度数'] >= 1].sample(n=5, random_state=42)
for _, row in sample.iterrows():
    text = row['评论'][:80] + '...' if len(row['评论']) > 80 else row['评论']
    print(f'\n评论: {text}')
    print(f'命中维度: {row["matched_dimensions"]}')
    print(f'命中关键词: {row["matched_words"]}')