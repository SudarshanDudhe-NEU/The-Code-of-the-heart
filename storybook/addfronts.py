from PIL import Image, ImageDraw, ImageFont


def add_text_to_image(
    image_path,
    text,
    position,
    font_path,
    font_size,
    font_color,
    stroke_width,
    stroke_fill,
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
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    # 显示图片
    image.show()

    # 保存图片
    image.save("storybook/output_image.png")


def add_text_with_shadow(
    image_path,
    text,
    position,
    font_path,
    font_size,
    text_color,
    stroke_width,
    stroke_fill,
    shadow_offset,
    shadow_color,
    shadow_opacity,
):
    # 打开图片
    image = Image.open(image_path).convert("RGBA")

    # 选择字体和大小
    font = ImageFont.truetype(font_path, font_size)

    # 创建一个可以在图片上绘图的对象
    draw = ImageDraw.Draw(image)

    # 计算阴影位置
    shadow_position = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])

    # 阴影颜色和透明度
    shadow_rgba = shadow_color + (int(255 * shadow_opacity),)  # 追加alpha通道

    # 首先绘制阴影
    draw.text(shadow_position, text, font=font, fill=shadow_rgba)

    # 再绘制文本
    draw.text(
        position,
        text,
        font=font,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    # 显示图片
    image.show()

    # 保存图片
    image.save("storybook/output_image.png")


# 使用函数
add_text_to_image(
    image_path="storybook/output.jpg",
    text="The Code of the",
    position=(200, 750),
    font_path="/Users/xingqi/Library/Fonts/华康手札体w5.ttf",  # 替换为您的字体文件的路径
    font_size=320,
    font_color="black",
    stroke_width=10,
    stroke_fill="white",
)

add_text_with_shadow(
    image_path="storybook/output_image.png",
    text="Heart",
    position=(813, 1350),
    font_path="/Users/xingqi/System/Library/Fonts/Supplemental/Sinhala MN.ttc",  # 替换为您的字体文件的路径
    font_size=360,
    text_color="white",
    stroke_width=10,
    stroke_fill="black",
    shadow_offset=(0, 70),
    shadow_color=(0, 0, 0),
    shadow_opacity=0.6,
)
