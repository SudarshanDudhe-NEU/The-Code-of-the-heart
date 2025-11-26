import textwrap
import cv2
from PIL import Image as pili, ImageDraw as pild, ImageFont as pilf
import numpy as np
import pandas as pd


def imgcompress_mem(img, k):
    width = int(img.shape[1] / k)
    height = int(img.shape[0] / k)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def cartoonize(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 17)
    bigedges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 5
    )
    cartoon = cv2.bitwise_and(img, img, mask=bigedges)
    return cartoon


# def wrap_text(text, font, max_width):
#     wrapped_text = []
#     text = str(text)
#     for line in text.split("\n"):
#         wrapped_text.extend(
#             textwrap.wrap(line, width=max_width, replace_whitespace=False)
#         )
#     return wrapped_text

def wrap_text(text, font, max_width):
    wrapped_text = []
    # 确保 text 是一个字符串
    text = str(text)
    for line in text.split("\n"):
        # 对于中文，我们可以在任何字符之间换行
        while len(line) > max_width:
            wrapped_text.append(line[:max_width])
            line = line[max_width:]
        wrapped_text.append(line)
    return wrapped_text



def add_text_to_image(img, text, font_path):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_size = 20  # 调整字体大小
    font = ImageFont.truetype(font_path, font_size)

    # 计算文本的宽度和高度
    text_width, text_height = draw.textsize(text, font=font)

    # 计算文本的位置
    x = (img_pil.width - text_width) / 2
    y = img_pil.height - text_height - 10  # 调整文本的垂直位置

    # 在图像上添加文本
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 使用函数
img = cv2.imread('path_to_your_image.jpg')  # 使用你的图像文件路径
font_path = 'path_to_your_font.ttf'  # 使用你的字体文件路径
text = '你的文本'

img_with_text = add_text_to_image(img, text, font_path)
cv2.imshow('Image with Text', img_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()

def correct_image_orientation(img_path):
    with pili.open(img_path) as image:
        exif = image._getexif()
        rotated = False
        if exif is not None:
            orientation_tag = 274
            if orientation_tag in exif:
                orientation = exif[orientation_tag]
                if orientation == 2:
                    image = image.transpose(pili.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    image = image.rotate(180)
                elif orientation == 4:
                    image = image.rotate(180).transpose(pili.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    image = image.rotate(-90, expand=True).transpose(pili.FLIP_LEFT_RIGHT)
                    rotated = True
                elif orientation == 6:
                    image = image.rotate(-90, expand=True)
                    rotated = True
                elif orientation == 7:
                    image = image.rotate(90, expand=True).transpose(pili.FLIP_LEFT_RIGHT)
                    rotated = True
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
                    rotated = True

        if rotated:
            width, height = image.size
            new_dimen = min(width, height)
            left = (width - new_dimen) / 2
            top = (height - new_dimen) / 2
            right = (width + new_dimen) / 2
            bottom = (height + new_dimen) / 2
            image = image.crop((left, top, right, bottom))

        corrected_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return corrected_img


def cartoonize_compress_and_text(
    input_path,
    k,
    text="",
    # font_path="/Users/xingqi/System/Library/Fonts/Supplemental/Arial.ttf",
    font_path="/Users/xingqi/System/Library/Fonts/Hiragino Sans GB.ttc"
    #
):
    img = correct_image_orientation(input_path)  # Correctly read and adjust image orientation
    cartoon_img = cartoonize(img)  # Using the corrected img
    compressed_cartoon = imgcompress_mem(cartoon_img, k)
    if text:
        compressed_cartoon = add_text_to_image(compressed_cartoon, text, k, font_path)
    return compressed_cartoon


def layout_images_to_pages(folder, image_names, texts, max_width_per_page=1700):
    imgs = [
        cartoonize_compress_and_text(folder + "/" + str(i) + ".jpg", 8, text)
        for i, text in zip(image_names, texts)
    ]

    target_height = 400
    resized_imgs = []
    white = [255, 255, 255]

    for img in imgs:
        original_height, original_width = img.shape[0], img.shape[1]
        new_width = int((target_height / original_height) * original_width)
        new_dim = (new_width, target_height)
        resized_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

        # Modified border sizes as per your request
        bordered_img = cv2.copyMakeBorder(
            resized_img, 20, 20, 10, 10, cv2.BORDER_CONSTANT, value=white
        )
        resized_imgs.append(bordered_img)

    pages = []
    current_page = []
    current_width = 0
    current_row = []

    for img in resized_imgs:
        if current_width + img.shape[1] <= max_width_per_page:
            current_row.append(img)
            current_width += img.shape[1]
        else:
            padding_width_each_side = (max_width_per_page - current_width) // 2
            padding_img_left = np.ones((target_height + 40, padding_width_each_side, 3), dtype=np.uint8) * 255
            padding_img_right = np.ones(
                (target_height + 40, max_width_per_page - current_width - padding_width_each_side, 3),
                dtype=np.uint8
            ) * 255

            current_row.insert(0, padding_img_left)
            current_row.append(padding_img_right)

            current_page.append(np.concatenate(current_row, axis=1))
            current_row = [img]
            current_width = img.shape[1]

        if len(current_row) >= 4:  # max 4 images per row
            if current_width < max_width_per_page:
                padding_width_each_side = (max_width_per_page - current_width) // 2
                padding_img_left = np.ones((target_height + 40, padding_width_each_side, 3), dtype=np.uint8) * 255
                padding_img_right = np.ones(
                    (target_height + 40, max_width_per_page - current_width - padding_width_each_side, 3),
                    dtype=np.uint8
                ) * 255

                current_row.insert(0, padding_img_left)
                current_row.append(padding_img_right)

            current_page.append(np.concatenate(current_row, axis=1))
            current_row = []
            current_width = 0

        if len(current_page) >= 5:  # max 5 rows per page
            pages.append(current_page)
            current_page = []

    if current_row:
        if current_width < max_width_per_page:
            padding_width_each_side = (max_width_per_page - current_width) // 2
            padding_img_left = np.ones((target_height + 40, padding_width_each_side, 3), dtype=np.uint8) * 255
            padding_img_right = np.ones(
                (target_height + 40, max_width_per_page - current_width - padding_width_each_side, 3),
                dtype=np.uint8
            ) * 255

            current_row.insert(0, padding_img_left)
            current_row.append(padding_img_right)

        current_page.append(np.concatenate(current_row, axis=1))

    if current_page:
        pages.append(current_page)

    output_paths = []
    for idx, page in enumerate(pages):
        merged_page = np.concatenate(page, axis=0)
        output_path = f"storybook/template/zh_merged_page_{idx}.png"
        cv2.imwrite(output_path, merged_page)
        output_paths.append(output_path)

    return output_paths

# Using the function
# output_pages = layout_images_to_pages(
#     "storybook/template/pic",
#     ["6", "7", "8", "9", "10"],
#     [
#         "Xingqi faces a technical challenge while fine-tuning his AI model",
#         "Sudarshan discusses a new type of chip that can efficiently power AI models",
#         "WALL-E(Xingxing) is activated for the first time and asks",
#         "Who am I? Where do I come from?",
#         "Hello!"
#     ],
# )

# for page in output_pages:
#     print(page)

def read_data_from_excel(excel_file_path):
    # 读取 Excel 文件
    data = pd.read_excel(excel_file_path, engine='openpyxl')

    # 假设第一列是图像名称，第二列是文本
    image_names = data.iloc[:, 0].tolist()
    texts = data.iloc[:, 1].tolist()

    return image_names, texts

image_names, texts = read_data_from_excel("storybook/template/source_zh.xlsx")

output_pages = layout_images_to_pages(
    "storybook/template/pic",
    image_names,
    texts,
)

for page in output_pages:
    print(page)
