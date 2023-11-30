from tkinter import N

input_test_file_path = "/root/renxueying/query_reacall/chinaLaw/recall/350/newquestion_350_recall"
output_prompt_file_path = "/root/renxueying/query_reacall/chinaLaw/recall/350/chinaLaw350Prompt.txt"

# input_test_file_path = "/root/projects/neutrino/langchain-ChatGLM/query/china_law/newquestion1_recall/query_eval_score_600_sentence_size_400"
# output_prompt_file_path = "/root/projects/neutrino/langchain-ChatGLM/query/china_law/newquestion1_recall/chinalLawPrompt.txt"

# PROMPT_TEMPLATE_dict = {
#     "cn":"""请根据我提供的问题和参考文献帮助我解决任务：
# 问题：{question}
# 参考文献：{context}
# 任务：请帮我判断这些文献与问题是否相关，将不相关的文献过滤掉，基于剩余相关的文献对此问题进行总结回答，并在相应位置标注引用的相关文献，若无相关文献则不需要再做总结回答。
# 输出格式：按照json格式输出,"相关文献有":[],"总结回答":[],且相关文献数字前都要统一加上reference
# """,
#     "en":"""Please help me solve the task based on the questions and references I have provided:
# Question: {question}
# Reference: {context}
# Task: Please help me to determine whether the literature is relevant to the question, filter out the irrelevant literature, summarise the answer to the question based on the remaining relevant literature, and mark the relevant literature cited in brackets [] in the appropriate place. If there is no relevant literature, there is no need to summarize and return to the "summary answer": [].
# Output format: according to the json format output, "relevant literature have":[], "summary answer":[], and relevant literature before the number of figures should be uniformly added reference
# """
# }
PROMPT_TEMPLATE_dict = {
    "cn":"""请根据我提供的问题和参考文献帮助我解决任务：
问题：{question}
参考文献：{context}
任务：请帮我判断哪些文献与问题相关，并列出相关文献的标号。
输出格式：请按照json格式返回结果，如："相关文献有":[1, 2, 3]。
""",
    "en":"""Please help me solve the task based on the questions and references I have provided:
Question: {question}
Reference: {context}
Task: Please help me to determine whether the literature is relevant to the question, filter out the irrelevant literature, summarise the answer to the question based on the remaining relevant literature, and mark the relevant literature cited in brackets [] in the appropriate place. If there is no relevant literature, there is no need to summarize and return to the "summary answer": [].
Output format: according to the json format output, "relevant literature have":[], "summary answer":[], and relevant literature before the number of figures should be uniformly added reference
"""
}

prompts = {}

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

# 读取召回文件
with open(input_test_file_path) as file:
    for line in file:
        parts = line.strip().split("\t")
        # print(parts)
        question = parts[0]
        recall = " ".join(parts[1:])

        if question in prompts:
            prompts[question].append(recall)
        else:
            prompts[question] = [recall]

# # 生成Prompt文件
with open(output_prompt_file_path, "w") as file:
    for question, recalls in prompts.items():
        context = "\n".join([f"reference{i+1}: {recall}" for i, recall in enumerate(recalls)])
        prompt_template = PROMPT_TEMPLATE_dict["en"]
        if is_contains_chinese(question):
            prompt_template = PROMPT_TEMPLATE_dict["cn"]
        prompt = prompt_template.format(question=question, context=context)
        file.write(prompt + "\n")
print("Prompt文件生成完成并保存到文件:", output_prompt_file_path)