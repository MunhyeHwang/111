from transformers import AutoModel, AutoTokenizer

model_name = "hfl/chinese-roberta-wwm-ext"

# 下载模型
model = AutoModel.from_pretrained(model_name)

# 下载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存到本地
save_path = "./chinese-roberta-wwm-ext"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("模型下载完成！")