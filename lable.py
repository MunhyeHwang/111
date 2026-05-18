import pandas as pd
df = pd.read_csv('huizong_cleaned.csv', encoding='gbk').rename(columns={'lable': 'label'})
print(df['label'].value_counts())
print(f'\n好评比例: {(df["label"]==1).sum()/len(df)*100:.2f}%')
print(f'差评比例: {(df["label"]==0).sum()/len(df)*100:.2f}%')
