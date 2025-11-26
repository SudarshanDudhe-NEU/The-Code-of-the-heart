import cv2
from PIL import Image as pili, ImageDraw as pild, ImageFont as pilf

def imgcompress_mem(img, k):
    # set the ratio of resized image
    width = int((img.shape[1])/k)
    height = int((img.shape[0])/k)
    # resize the image by resize() function of openCV library
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def cartoonize_custom(input_img, output_path, text="", nlines=1, font="verdana"):
    # 增加双边滤波，有助于保持边缘清晰同时进行模糊处理
    img = cv2.bilateralFilter(input_img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 17)
    bigedges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 5)
    cartoon = cv2.bitwise_and(img, img, mask=bigedges)

    # 如果没有文字，直接保存和返回
    if not text:
        cv2.imwrite(output_path, cartoon)
        return cartoon

    # 添加文字
    font_size_mapping = {16: 24, 14: 18, 12: 18, 8: 20}
    k = int(img.shape[1] / cartoon.shape[1])  # 计算压缩因子
    font_size = font_size_mapping.get(k, 82)
    cartoon_pil = pili.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    draw = pild.Draw(cartoon_pil)
    myfont = pilf.truetype(font, font_size)
    bbox = myfont.getbbox(text)
    h = bbox[3] - bbox[1]
    x, y = 10, cartoon_pil.height - nlines * h - 10
    draw.text((x, y), text, fill=(248,248,248), font=myfont)

    # 保存图片并返回
    cartoon_pil.save(output_path)
    return cartoon

# 使用函数
img = cv2.imread('storybook/template/10.jpg', cv2.IMREAD_UNCHANGED)
compressed_img = imgcompress_mem(img, 12)  # 假设k=12
cartoonize_custom(compressed_img, 'storybook/template/output.jpg', "My Text Here", 1, "/Users/xingqi/System/Library/Fonts/Helvetica.ttc")
