from PIL import Image, ImageDraw, ImageFont


def add_rectangle(image_path, rectangle_position, rectangle_color):
    # 打开图片
    image = Image.open(image_path)

    # 创建一个可以在图片上绘图的对象
    draw = ImageDraw.Draw(image)

    # 添加矩形
    draw.rectangle(rectangle_position, fill=rectangle_color)

    # 保存图片
    image.save("storybook/output_with_rectangle.png")

def add_text_to_image(
    image_path,
    text,
    position,
    font_path,
    font_size,
    font_color,
):
    # 打开图片
    image = Image.open(image_path)

    # 选择字体和大小
    font = ImageFont.truetype(font_path, font_size)

    # 创建一个可以在图片上绘图的对象
    draw = ImageDraw.Draw(image)

    # 在图片上添加文字
    draw.text(
        position,
        text,
        font=font,
        fill=font_color,
    )

    # 保存图片
    image.save("storybook/output_final.png")

# 使用函数
add_rectangle(
    image_path="storybook/output_image.png",
    rectangle_position=(0, 3661, 2591, 3961),  # (左上角x, 左上角y, 右下角x, 右下角y)
    rectangle_color=(255, 255, 255),  # RGB: 白色不透明矩形
)

add_text_to_image(
    image_path="storybook/output_with_rectangle.png",
    text="Xingqi Li",
    position=(133, 3750),
    font_path="/Users/xingqi/Library/Fonts/华康翩翩体W5.ttf",  # 替换为您的字体文件的路径
    font_size=80,
    font_color="black",
)

add_text_to_image(
    image_path="storybook/output_final.png",
    text="Sudarshan Dudhe",
    position=(600, 3750),
    font_path="/Users/xingqi/Library/Fonts/华康翩翩体W5.ttf",  # 替换为您的字体文件的路径
    font_size=80,
    font_color="black",
)
add_text_to_image(
    image_path="storybook/output_final.png",
    text="Aniruddh Goudar",
    position=(1300, 3750),
    font_path="/Users/xingqi/Library/Fonts/华康翩翩体W5.ttf",  # 替换为您的字体文件的路径
    font_size=80,
    font_color="black",
)
add_text_to_image(
    image_path="storybook/output_final.png",
    text="Xingxing Xiao",
    position=(2050, 3750),
    font_path="/Users/xingqi/Library/Fonts/华康翩翩体W5.ttf",  # 替换为您的字体文件的路径
    font_size=80,
    font_color="black",
)
