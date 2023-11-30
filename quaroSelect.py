import random
import json

# # 读取原始文件
# with open("/root/renxueying/long_article/recall/atricle1102.jsonl", "r") as file:
#     lines = file.readlines()

# # 计算抽取的行数
# sample_size = 200
# print("sample_size==========",sample_size)
# # 随机抽取行索引
# random_indices = random.sample(range(len(lines)), sample_size)

# # 抽取行并保存到新文件
# with open("/root/renxueying/long_article/recall/atricle1107_200.jsonl", "w") as new_file:
#     for index in random_indices:
#         new_file.write(lines[index])


# # 读取文件并筛选信息行
# selected_lines = []
# with open("/root/projects/neutrino/langchain-ChatGLM/query/quora.jsonl", "r") as file:
#     for line in file:
#         data = json.loads(line)
#         text = data.get("text", "")
#         if any(keyword in text for keyword in ["machine learning","CNN", "RNN", "SVM", "embedding","Back Propagation"]):
#             selected_lines.append(line)
# # 随机挑选300条数据
# random_selection = random.sample(selected_lines, 300)
# # 将筛选结果写入新文件
# with open("quora300.jsonl", "w") as output_file:
#     for line in selected_lines:
#         output_file.write(line)
# print("保留的行数：", len(selected_lines))

# 定义关键词列表
keywords = ["machine learning", "CNN", "RNN", "SVM", "embedding", "Back Propagation","NLP","batch","classification","clustering","convergence","matrix","convolution"]
# keywords = ["law","litigation","compensation"]
# 读取文件并筛选信息行
selected_lines = []
with open("/root/renxueying/query_reacall/quaro.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)
        text = data.get("text", "")
        if any(keyword in text for keyword in keywords):
            selected_lines.append(line)

# # 随机挑选300条数据
# random_selection = random.sample(selected_lines,70)

# 将挑选结果写入新文件
with open("/root/renxueying/query_reacall/ml/MLquery.jsonl", "w") as output_file:
    for line in selected_lines:
        output_file.write(line)

print("保留的行数：", len(selected_lines))