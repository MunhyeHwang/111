import re
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# =========================
# 0. 中文字体设置
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
font_path = r'C:\Windows\Fonts\simhei.ttf'   # 如果没有就换成 msyh.ttc

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv('huizong_cleaned.csv', encoding='gbk').rename(columns={'lable': 'label'})
df = df[['评论', 'label']].dropna().copy()
df['评论'] = df['评论'].astype(str)

# =========================
# 2. 读取停用词
# =========================
with open('hit_stopwords.txt', encoding='ANSI') as f:
    stopwords = set(f.read().split())

# 建议把明显的场景词放进停用词，避免污染情绪词云
stopwords |= {
    '京东', '护士', '服务', '上门', '感觉', '真的', '非常', '还是', '就是',
    '这次', '这个', '一个', '可以', '比较', '觉得', '已经', '没有',
    '我们', '你们', '他们', '进行', '提供', '情况', '时候', '一下'
}

# =========================
# 3. 情绪词典
# 可自行继续扩充
# =========================
pos_words = {
    '满意','很好','不错','推荐','方便','专业','耐心','细心','贴心','认真','负责',
    '及时','高效','周到','热情','温柔','舒适','放心','准时','熟练','干净','快捷',
    '满意度高','态度好','很快','放心了','特别好','值得','赞','点赞'
}

neg_words = {
    '差','不好','失望','糟糕','慢','很慢','太慢','问题','麻烦','投诉','错误','失败',
    '不专业','不满意','不方便','粗心','敷衍','延迟','等待','担心','不准','难受',
    '态度差','不舒服','痛','贵','退款','取消','漏掉','耽误','折腾','不行','崩溃'
}

# =========================
# 4. 清洗 + jieba 分词
# =========================
clean_text = lambda s: re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]+', ' ', s)

tokenize = lambda s: [
    w for w in jieba.lcut(clean_text(s))
    if len(w) > 1 and w not in stopwords and not w.isdigit()
]

df['tokens'] = df['评论'].map(tokenize)
df['text_cut'] = df['tokens'].map(lambda x: ' '.join(x))
#保存
with open('segmented_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(df['text_cut'].tolist()))
df = df[df['text_cut'].str.strip() != ''].copy()

# =========================
# 5. TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.85
)
X = vectorizer.fit_transform(df['text_cut'])

tfidf_scores = X.mean(axis=0).A1
top_idx = tfidf_scores.argsort()[-20:][::-1]
top_terms = vectorizer.get_feature_names_out()
print('\nTF-IDF前20词汇及均值:')
for i, idx in enumerate(top_idx, 1):
    print(f"{i:2d}. {top_terms[idx]}: {tfidf_scores[idx]:.6f}")

# =========================
# 6. KMeans 聚类
# =========================
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
df['cluster'] = kmeans.fit_predict(X)

# 7. 用原始标签映射正负簇
cluster_sentiment = df.groupby('cluster')['label'].mean().sort_values()
neg_cluster, pos_cluster = cluster_sentiment.index[0], cluster_sentiment.index[1]

df['cluster_name'] = df['cluster'].map({
    pos_cluster: '正面簇',
    neg_cluster: '负面簇'
})

# =========================
# 8. 给每条评论计算情绪分数
# 目的是在簇内二次筛选，让负面簇更“纯”
# =========================
count_sentiment = lambda words, lexicon: sum(w in lexicon for w in words)

df['pos_cnt'] = df['tokens'].map(lambda x: count_sentiment(x, pos_words))
df['neg_cnt'] = df['tokens'].map(lambda x: count_sentiment(x, neg_words))
df['sent_score'] = df['pos_cnt'] - df['neg_cnt']

# 只保留更“像正面/负面”的评论来做词云
pos_df = df[(df['cluster_name'] == '正面簇') & (df['sent_score'] >= 0)].copy()
neg_df = df[(df['cluster_name'] == '负面簇') & (df['sent_score'] <= 0)].copy()

# 如果筛选后太少，可以放宽阈值
if len(pos_df) < 50:
    pos_df = df[df['cluster_name'] == '正面簇'].copy()

if len(neg_df) < 50:
    neg_df = df[df['cluster_name'] == '负面簇'].copy()

# 9. 词频统计
def get_word_counter(dataframe):
    return Counter([w for row in dataframe['tokens'] for w in row])

def get_sentiment_counter(dataframe, lexicon):
    return Counter([w for row in dataframe['tokens'] for w in row if w in lexicon])

# 更丰富的词云：取更多词
pos_counter_all = get_word_counter(pos_df)
neg_counter_all = get_word_counter(neg_df)

# 更准确的情绪词云：只保留情绪词
pos_counter_sent = get_sentiment_counter(pos_df, pos_words)
neg_counter_sent = get_sentiment_counter(neg_df, neg_words)

# 采用折中：优先情绪词，不够时补充全部高频词
def merge_counter(main_counter, backup_counter, topn=150):
    result = Counter(main_counter)
    for w, c in backup_counter.most_common(topn):
        result[w] += c
    return Counter(dict(result.most_common(topn)))

pos_counter = merge_counter(pos_counter_sent, pos_counter_all, topn=150)
neg_counter = merge_counter(neg_counter_sent, neg_counter_all, topn=150)

pos_top = pos_counter.most_common(15)
neg_top = neg_counter.most_common(15)

# 10. 聚类中心关键词
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
cluster_keywords = {
    i: [terms[ind] for ind in order_centroids[i, :100]]
    for i in range(2)
}

print('各簇原始标签均值：')
print(cluster_sentiment)

print('\n正面簇前100关键词：')
print(cluster_keywords[pos_cluster])

print('\n负面簇前100关键词：')
print(cluster_keywords[neg_cluster])

print('\n正面簇前20关键西及TF-IDF：')
print(pos_top[:20])

print('\n负面簇前20关键西及TF-IDF：')
print(neg_top[:20])

# 11. 词云图
def make_wordcloud(counter_obj, out_file):
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        width=1600,
        height=900,
        max_words=200,
        collocations=False
    ).generate_from_frequencies(dict(counter_obj))
    wc.to_file(out_file)

make_wordcloud(pos_counter, 'positive_wordcloud.png')
make_wordcloud(neg_counter, 'negative_wordcloud.png')

# 12. 柱状图
def plot_bar(word_counts,title,out_file):
    words, counts = zip(*word_counts)
    plt.figure(figsize=(14, 7))
    plt.bar(words, counts, color='#85CCCD')
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.close()

plot_bar(pos_top, '正面簇高频词柱状图', 'positive_bar.png')
plot_bar(neg_top, '负面簇高频词柱状图', 'negative_bar.png')

# 13. 样本量分布
cluster_counts = df['cluster_name'].value_counts()

plt.figure(figsize=(14, 6))
plt.barh(cluster_counts.index, cluster_counts.values,
         color='#C6E6E9', height=0.3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('cluster_size.png', dpi=300)
plt.close()

# 14. 保存结果
summary = df.groupby(['cluster', 'cluster_name']).agg(
    样本量=('评论', 'count'),
    原始标签均值=('label', 'mean'),
    平均正向词数=('pos_cnt', 'mean'),
    平均负向词数=('neg_cnt', 'mean')
).reset_index()

summary.to_csv('cluster_summary.csv', index=False, encoding='utf-8-sig')
df.to_csv('clustered_comments.csv', index=False, encoding='utf-8-sig')

print('\n文件已保存：')
print('clustered_comments.csv')
print('cluster_summary.csv')
print('positive_wordcloud.png')
print('negative_wordcloud.png')
print('positive_bar.png')
print('negative_bar.png')
print('cluster_size.png')