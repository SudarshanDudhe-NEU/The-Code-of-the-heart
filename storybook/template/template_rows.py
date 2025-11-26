import textwrap
import cv2
from PIL import Image as pili, ImageDraw as pild, ImageFont as pilf
import numpy as np
from IPython.display import Image, display


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


def wrap_text(text, font, max_width):
    """
    Wrap the text to fit within the specified width.
    """
    wrapped_text = []
    for line in text.split("\n"):
        wrapped_text.extend(
            textwrap.wrap(line, width=max_width, replace_whitespace=False)
        )
    return wrapped_text


def add_text_to_image(img, text, k, font_path):
    font_size_mapping = {16: 24, 14: 18, 12: 18, 8: 24}
    font_size = font_size_mapping.get(k, 82)
    img_pil = pili.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = pild.Draw(img_pil)
    myfont = pilf.truetype(font_path, font_size)

    # Wrap the text based on the image width and font size in pixel
    max_line_length = int(img_pil.width / (font_size / 2))
    wrapped_text_lines = wrap_text(text, myfont, max_line_length)

    # Calculate total height of the wrapped text
    total_text_height = (
        len(wrapped_text_lines) * font_size
        + (len(wrapped_text_lines) - 1) * font_size * 0.2
    )  # adding 20% spacing between lines

    # Drawing text starts here
    y = (
        img_pil.height - total_text_height - font_size * 0.5
    )  # added half the font size as extra padding
    for line in wrapped_text_lines:
        bbox = draw.textbbox((0, 0), line, font=myfont)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x = (img_pil.width - width) / 2
        draw.text(
            (x, y), line, fill="white", font=myfont, stroke_width=1, stroke_fill="black"
        )
        y += height + font_size * 0.2  # adding 20% spacing between lines

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cartoonize_compress_and_text(
    input_path,
    k,
    text="",
    font_path="/Users/xingqi/System/Library/Fonts/Supplemental/Arial.ttf",
):
    img = correct_image_orientation(input_path)  # 正确地读取并纠正图像方向

    cartoon_img = cartoonize(img)  # 使用修正的 img
    compressed_cartoon = imgcompress_mem(cartoon_img, k)
    if text:
        compressed_cartoon = add_text_to_image(compressed_cartoon, text, k, font_path)
    return compressed_cartoon


# 使用函数
# cartoonize_compress_and_text(
#     "storybook/template/10.jpg",
#     "storybook/template/output_final.jpg",
#     12,
#     "Xingqi faces a technical challenge while fine-tuning his AI model in his laboratory in Beijing.",
#     "/Users/xingqi/System/Library/Fonts/Supplemental/Arial.ttf",
# )


def correct_image_orientation(img_path):
    with pili.open(img_path) as image:
        exif = image._getexif()
        rotated = False
        if exif is not None:
            orientation_tag = 274  # The tag which contains orientation data
            if orientation_tag in exif:
                orientation = exif[orientation_tag]
                if orientation == 2:
                    image = image.transpose(pili.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    image = image.rotate(180)
                elif orientation == 4:
                    image = image.rotate(180).transpose(pili.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    image = image.rotate(-90, expand=True).transpose(
                        pili.FLIP_LEFT_RIGHT
                    )
                    rotated = True
                elif orientation == 6:
                    image = image.rotate(-90, expand=True)
                    rotated = True
                elif orientation == 7:
                    image = image.rotate(90, expand=True).transpose(
                        pili.FLIP_LEFT_RIGHT
                    )
                    rotated = True
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
                    rotated = True

        if rotated:  # If the image was rotated, let's crop it to a square
            width, height = image.size
            new_dimen = min(width, height)
            left = (width - new_dimen) / 2
            top = (height - new_dimen) / 2
            right = (width + new_dimen) / 2
            bottom = (height + new_dimen) / 2
            image = image.crop((left, top, right, bottom))

        corrected_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return corrected_img


def layout_images_to_pages(folder, image_names, texts, max_width_per_page=2000):
    imgs = [
        cartoonize_compress_and_text(folder + "/" + i + ".jpg", 8, text)
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
        bordered_img = cv2.copyMakeBorder(
            resized_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=white
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
            padding_img_left = (
                np.ones(
                    (target_height + 10, padding_width_each_side, 3), dtype=np.uint8
                )
                * 255
            )
            padding_img_right = (
                np.ones(
                    (
                        target_height + 10,
                        max_width_per_page - current_width - padding_width_each_side,
                        3,
                    ),
                    dtype=np.uint8,
                )
                * 255
            )

            current_row.insert(0, padding_img_left)
            current_row.append(padding_img_right)

            current_page.append(np.concatenate(current_row, axis=1))
            current_row = [img]
            current_width = img.shape[1]

        if len(current_row) >= 4:  # max 4 images per row
            if current_width < max_width_per_page:
                padding_width_each_side = (max_width_per_page - current_width) // 2
                padding_img_left = (
                    np.ones(
                        (target_height + 10, padding_width_each_side, 3), dtype=np.uint8
                    )
                    * 255
                )
                padding_img_right = (
                    np.ones(
                        (
                            target_height + 10,
                            max_width_per_page
                            - current_width
                            - padding_width_each_side,
                            3,
                        ),
                        dtype=np.uint8,
                    )
                    * 255
                )

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
            padding_img_left = (
                np.ones(
                    (target_height + 10, padding_width_each_side, 3), dtype=np.uint8
                )
                * 255
            )
            padding_img_right = (
                np.ones(
                    (
                        target_height + 10,
                        max_width_per_page - current_width - padding_width_each_side,
                        3,
                    ),
                    dtype=np.uint8,
                )
                * 255
            )

            current_row.insert(0, padding_img_left)
            current_row.append(padding_img_right)

        current_page.append(np.concatenate(current_row, axis=1))

    if current_page:
        pages.append(current_page)

    output_paths = []

    for idx, page in enumerate(pages):
        merged_page = np.concatenate(page, axis=0)
        output_path = f"storybook/template/merged_page_{idx}.png"
        cv2.imwrite(output_path, merged_page)
        output_paths.append(output_path)

    return output_paths


# 使用函数
output_pages = layout_images_to_pages(
    "storybook/template/pic",
    ["6", "7", "8", "9", "10"],
    [
        "Xingqi faces a technical challenge while fine-tuning his AI model",
        "Sudarshan discusses a new type of chip that can efficiently power AI models ",
        "WALL-E(Xingxing) is activated for the first time and asks",
        "Who am I? Where do I come from?",
        "Hello!",
    ],
)

for page in output_pages:
    print(page)
