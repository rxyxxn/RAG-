from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os


def convert_pdf_to_txt(pdf_path, txt_path):
    # 打开PDF文件
    with open(pdf_path, 'rb') as file:
        # 创建一个PDFResourceManager对象
        resource_manager = PDFResourceManager()
        # 创建一个StringIO对象，用于存储提取的文本内容
        output = StringIO()

        # 创建一个TextConverter对象
        converter = TextConverter(resource_manager, output, laparams=LAParams())

        # 创建一个PDFPageInterpreter对象
        interpreter = PDFPageInterpreter(resource_manager, converter)

        # 逐页解析文档
        for page in PDFPage.get_pages(file):
            interpreter.process_page(page)

        # 获取提取的文本内容
        text = output.getvalue().replace("\n","").replace(" ","")
        output.close()
        file.close()
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
            txt_file.close()


# 指定包含PDF文件的文件夹
pdf_folder = '/root/renxueying/history/test/history'


# 指定要保存TXT文件的文件夹
txt_folder = '/root/renxueying/history/test/historydeal'

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
