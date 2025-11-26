from PIL import Image

def png_to_pdf(png_path, output_path):
    # 打开 PNG 图像
    image = Image.open(png_path)

    # 如果图像有 alpha 通道（透明度），则将其转换为 RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # 保存为 PDF
    image.save(output_path, 'PDF')

# 使用函数
png_to_pdf('storybook/output_final.png', 'storybook/output_final.pdf')
