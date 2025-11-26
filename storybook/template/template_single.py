import cv2
from PIL import Image as pili, ImageDraw as pild, ImageFont as pilf
import numpy as np
import textwrap


def imgcompress_mem(img, k):
    width = int(img.shape[1] / k)
    height = int(img.shape[0] / k)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def cartoonize(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
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
    font_size_mapping = {16: 24, 14: 18, 12: 18, 8: 20}
    font_size = font_size_mapping.get(k, 82)
    img_pil = pili.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = pild.Draw(img_pil)
    myfont = pilf.truetype(font_path, font_size)

    # Wrap the text based on the image width and font size
    max_line_length = int(img_pil.width / (font_size / 2))
    wrapped_text_lines = wrap_text(text, myfont, max_line_length)

    # Calculate total height of the wrapped text
    total_text_height = (
        len(wrapped_text_lines) * font_size
        + (len(wrapped_text_lines) - 1) * font_size * 0.2
    )  # adding 20% spacing between lines

    # Start drawing from the bottom of the image, offset by total text height plus some extra padding
    y = (
        img_pil.height - total_text_height - font_size * 0.5
    )  # added half the font size as extra padding

    # Draw each line of the wrapped text
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
    output_path,
    k,
    text="",
    font_path="/Users/xingqi/System/Library/Fonts/Helvetica.ttc",
):
    cartoon_img = cartoonize(input_path)
    compressed_cartoon = imgcompress_mem(cartoon_img, k)
    if text:
        compressed_cartoon = add_text_to_image(compressed_cartoon, text, k, font_path)
    cv2.imwrite(output_path, compressed_cartoon)


# 使用函数
cartoonize_compress_and_text(
    "storybook/template/8.jpg",
    "storybook/template/output_final2.jpg",
    12,
    "Xingqi faces a technical challenge while fine-tuning his AI model in his laboratory in Beijing.",
    "/Users/xingqi/System/Library/Fonts/Supplemental/Arial.ttf",
)
