# import os
# import PyPDF2

# # 定义一个函数，用于将PDF文件转换为文本文件
# def convert_pdf_to_txt(pdf_path, txt_path):
#     with open(pdf_path, 'rb') as pdf_file:
#         pdf_reader = PyPDF2.PdfFileReader(pdf_file)
#         text = ''
#         for page_num in range(pdf_reader.numPages):
#             page = pdf_reader.getPage(page_num)
#             text += page.extractText()
        
#         with open(txt_path, 'w', encoding='utf-8') as txt_file:
#             txt_file.write(text)

# # 指定包含PDF文件的文件夹
# pdf_folder = '/root/history1'

# # 指定要保存TXT文件的文件夹
# txt_folder = '/root/renxueying/history/historyDeal'
# os.makedirs(txt_folder, exist_ok=True)

# # 遍历文件夹中的PDF文件
# for filename in os.listdir(pdf_folder):
#     if filename.endswith('.pdf'):
#         pdf_path = os.path.join(pdf_folder, filename)
#         txt_filename = os.path.splitext(filename)[0] + '.txt'
#         txt_path = os.path.join(txt_folder, txt_filename)
        
#         # 将PDF转换为TXT
#         convert_pdf_to_txt(pdf_path, txt_path)
#         print(f'转换完成: {pdf_path} -> {txt_path}')
import os
from PyPDF2 import PdfReader
import re
# 定义一个函数，用于将PDF文件转换为文本文件
def convert_pdf_to_txt(pdf_path, txt_path):
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # # 忽略无法编码的字符，然后再进行解码
    # text = text.encode('utf-8', 'ignore').decode('utf-8')
    
    # 替换无法编码的字符为指定的字符（例如空白字符）
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

# def convert_pdf_to_txt(pdf_path, txt_path):
#     pdf_reader = PdfReader(pdf_path)
#     text = ''
#     for page in pdf_reader.pages:
#         text += page.extract_text()
    
#     with open(txt_path, 'w', encoding='utf-8') as txt_file:
#         txt_file.write(text)

# 指定包含PDF文件的文件夹
pdf_folder = '/root/renxueying/history/test/history'

# 指定要保存TXT文件的文件夹
txt_folder = '/root/renxueying/history/historyDeal'
os.makedirs(txt_folder, exist_ok=True)

# 遍历文件夹中的PDF文件
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(txt_folder, txt_filename)
        
        # 将PDF转换为TXT
        convert_pdf_to_txt(pdf_path, txt_path)
        print(f'转换完成: {pdf_path} -> {txt_path}')
