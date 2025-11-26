from PIL import Image
from reportlab.pdfgen import canvas

def pngs_to_pdf(png_files, output_file):
    # 获取第一个图像的尺寸以确定PDF的大小
    img = Image.open(png_files[0])
    img_width, img_height = img.size

    # 使用默认的DPI（72 DPI）将像素尺寸转换为点
    pdf_width = img_width
    pdf_height = img_height

    # 在循环外部创建canvas对象
    c = canvas.Canvas(output_file, pagesize=(pdf_width, pdf_height))

    for png_file in png_files:
        c.drawImage(png_file, 0, 0, width=pdf_width, height=pdf_height)
        # 在每个图像之后开始一个新的页面
        c.showPage()

    # 保存PDF
    c.save()

# 用你的 PNG 文件名替换以下列表
png_files = [
    "storybook/template/merged_page_0.png",
    "storybook/template/merged_page_1.png",
    "storybook/template/merged_page_2.png",
    "storybook/template/merged_page_3.png",
    "storybook/template/merged_page_4.png",
    "storybook/template/merged_page_5.png",
]
output_file = "storybook/template/output.pdf"
pngs_to_pdf(png_files, output_file)
