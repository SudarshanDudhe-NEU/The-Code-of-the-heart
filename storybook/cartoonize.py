import cv2
from IPython.display import Image, display
import numpy as np

def cartoonize(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    # 增加双边滤波，有助于保持边缘清晰同时进行模糊处理
    img = cv2.bilateralFilter(img, 9, 75, 75)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 增加滤波器大小，进行更大程度的模糊处理
    gray_blur = cv2.medianBlur(gray, 17)

    # 增加块大小和C值，减少不必要的边缘
    bigedges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 5)

    cartoon = cv2.bitwise_and(img, img, mask=bigedges)

    # 如果需要显示图片，可以使用如下代码
    # cartoon_pil = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    # display(Image.fromarray(cartoon_pil))

    # 保存输出图片
    cv2.imwrite(output_path, cartoon)

# 使用函数
cartoonize('storybook/IMG.jpg', 'storybook/output.jpg')
