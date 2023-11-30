from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
import numpy as np
from typing import List, Tuple, Dict
from utils import torch_gc
from langchain.docstore.document import Document
import json
import os
from typing import (
    Optional,
    Any
)
embedding_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4,
        filter: Optional[Dict[str, Any]] = None, fetch_k: int = 20,
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)
    #print ("store_len", store_len)
    #print (indices)
    origin_docs = []
    for j, i in enumerate(indices[0]):
        #print (j, i)
        if i == -1 or 0 < self.score_threshold < scores[0][j]:
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i]
        #print ("id", _id)
        doc = self.docstore.search(_id)

        #也要返回origin doc
        origin_doc = self.docstore.search(_id)
        origin_doc.metadata["score"] = int(scores[0][j])
        origin_docs.append(origin_doc)

        #print ("doc", doc)
        if not self.chunk_conent:
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc.metadata["score"] = int(scores[0][j])
            docs.append(doc)
            continue

            
        #print ("wtf")
        id_set.add(i)
        docs_len = len(doc.page_content)
        #print (max(i, store_len - i))
        for k in range(1, max(i, store_len - i)):
            #print ("k", k)
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        break_flag = True
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
            if break_flag:
                break
    #print ("wtf??", len(origin_docs))
    if not self.chunk_conent:
        return docs, origin_docs
    #print ("id_set", id_set)
    if len(id_set) == 0 and self.score_threshold > 0:
        return [], []
    id_list = sorted(list(id_set))
    id_lists = seperate_list(id_list)
    """
    for doc in docs:
        print ("before page_content", doc.page_content, doc.metadata)
    """
    for id_seq in id_lists:
        for id in id_seq:
            if id == id_seq[0]:
                _id = self.index_to_docstore_id[id]
                doc = self.docstore.search(_id)
            else:
                _id0 = self.index_to_docstore_id[id]
                doc0 = self.docstore.search(_id0)
                doc.page_content += " " + doc0.page_content
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        doc.metadata["score"] = int(doc_score)
        docs.append(doc)
    """
    for doc in docs:
        print ("after page_content", doc.page_content, doc.metadata)
    """
    
    torch_gc()
    return docs, origin_docs

class RecallEval:
    def __init__(self, embedding_model_path, \
        embedding_device, chunk_size, chunk_conent, score_threshold, top_k):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path, model_kwargs={'device': embedding_device})
        # 匹配后单段上下文长度
        self.chunk_size = chunk_size
        self.chunk_conent = chunk_conent
        self.score_threshold = score_threshold
        self.top_k = top_k



    def get_knowledge_based_answer(self, query, vs_path):
        #把向量库的数据load进来
        #print ("self.embeddings ", self.embeddings)
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        #query转成embedding，去向量库检索 topk 
        related_docs_with_score, origin_docs = vector_store.similarity_search_with_score(query, k=self.top_k)
        torch_gc()
        result = []

        origin_doc_dict = {}
        for doc in origin_docs:
            origin_doc_dict[doc.metadata["source"]] = doc.page_content

        for doc in related_docs_with_score:
            origin_doc_content = origin_doc_dict[doc.metadata["source"]]
            #print ("doc", doc.page_content)
            #print ("origin_doc", origin_doc_content)
            #print (doc)
            result.append(doc.page_content + "\t" + origin_doc_content + "\t" + str(doc.metadata['score']))
            # result.append(str(doc.metadata['score'])+"{" + doc.metadata["source"] + "}")
            # result.append("score:" + str(doc.metadata['score']) + "{" + doc.metadata["source"] + "}")


        return result

# sentence_size_list = [200, 300, 00]
sentence_size_list = [400]
# score_list = [0, 200, 300, 350, 400, 600, 700]
score_list = [600]
# 存储所有召回结果的列表
all_recall_lists = []
for sentence_size in sentence_size_list:
    for score in score_list:
        recall_eval = RecallEval(
            embedding_model_path = "/root/projects/neutrino/langchain-ChatGLM/embeddding_models/GanymedeNil/text2vec-large-chinese",
            embedding_device = embedding_device,
            # 匹配后单段上下文长度
            chunk_size = 1000,
            chunk_conent = True,
            #chunk_conent = False,
            # 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，建议值为500，现在我们评估的都是0分
            score_threshold = score,
            top_k = 5
        )
        # 遍历所有文件夹
        vector_folder = "/root/projects/neutrino/langchain-ChatGLM/vector_store_with_year/ML"

        with open("/root/renxueying/query_reacall/ml/yuliang/350/mlquery350.json") as file, \
            open("/root/renxueying/query_reacall/ml/recall/350/mlquery350_recall", "w") as file2:
            for line in file:
                json_obj = json.loads(line.rstrip())
                origin_query = json_obj["text"]
                result_list = []
                for folder_name in os.listdir(vector_folder):
                    folder_path = os.path.join(vector_folder, folder_name)
                    recall_list = recall_eval.get_knowledge_based_answer(origin_query,folder_path)
                    # all_recall_lists = all_recall_lists.append(recall_list)
                    result_list.extend(recall_list)
                
                # Sort by 'doc.metadata['score']' and get the top 5 results
                sorted_results = sorted(result_list, key=lambda x: float(x.split(':')[1].split('{')[0]), reverse=True)
                top_five_results = sorted_results[:3]
                
                for recall in top_five_results:
                    file2.write(origin_query + "\t" + recall + "\n")  

        
# class Document:
#     def __init__(self, question, recall_result, relevance):
#         self.question = question
#         self.recall_result = recall_result
#         self.relevance = relevance

# def generate_prompt(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#     documents = []
#     for line in lines:
#         fields = line.strip().split('\t')
#         question = fields[0]
#         recall_result = fields[1]
#         relevance = fields[2]
#         document = Document(question, recall_result, relevance)
#         documents.append(document)

#     context = "\n".join([f"Question: {doc.question}\nRecall Result: {doc.recall_result}\nRelevance: {doc.relevance}" for doc in documents])
#     prompt = f"The following are the question, recall result, and relevance for each document:\n\n{context}"
    
#     return prompt

# file_path = "/root/projects/neutrino/langchain-ChatGLM/query/machine_learning/filtered_quora/machine_learning_query_eval_score_600_sentence_size_400"
# prompt = generate_prompt(file_path)
# prompt_path = "/root/projects/neutrino/langchain-ChatGLM/query/machine_learning/filtered_quora/prompt.txt"  # 指定要保存的文件路径

# # 将 prompt 写入文件
# with open(prompt_path, 'w') as file:
#     file.write(prompt)

# print("Prompt 已保存到文件:", prompt_path)

# recall_file_path = "/root/projects/neutrino/langchain-ChatGLM/query/machine_learning/eval_v2_keywords/machine_learning_query_eval_score_300_sentence_size_300"
# output_prompt_file_path = "/root/projects/neutrino/langchain-ChatGLM/query/machine_learning/prompt_file.txt"

# prompts = {}

# # 读取召回文件
# with open(recall_file_path) as file:
#     for line in file:
#         parts = line.strip().split("\t")
#         question = parts[0]
#         recall = parts[1]
#         print("recall======")
#         print(parts[1])

#         if question in prompts:
#             prompts[question].append(recall)
#         else:
#             prompts[question] = [recall]

# # 生成Prompt文件
# with open(output_prompt_file_path, "w") as file:
#     for question, recalls in prompts.items():
#         prompt = {
#             "question": question,
#             "recall": "\n".join(recalls)
#         }
#         file.write(json.dumps(prompt) + "\n")

# print("Prompt文件生成完成并保存到文件:", output_prompt_file_path)