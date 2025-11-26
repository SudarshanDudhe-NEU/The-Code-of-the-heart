import cv2
from IPython.display import Image, display
from PIL import Image

def cartoonize(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # 使用高斯双边滤波进行降噪
    img = cv2.bilateralFilter(img, 9, 90, 90)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 17)

    # 使用Canny边缘检测增强边缘
    edges = cv2.Canny(gray_blur, 100, 200)

    # 增强边缘效果
    bigedges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 5)

    cartoon = cv2.bitwise_and(img, img, mask=bigedges)

    cartoon_pil = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(cartoon_pil))

    cv2.imwrite(output_path, cartoon)

cartoonize('storybook/IMG-7111.jpg', 'storybook/output3.jpg')
