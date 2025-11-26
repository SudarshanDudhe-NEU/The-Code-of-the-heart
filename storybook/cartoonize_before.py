import cv2
import numpy as np

def cartoonize(image_path, output_path):
    # 读取图片
    img = cv2.imread(image_path)
    # 转换到灰度模式
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用中值滤波进行模糊处理
    gray_blur = cv2.medianBlur(gray, 5)
    # 使用自适应阈值检测边缘
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # 对原图像应用双边滤波
    color = cv2.bilateralFilter(img, 9, 300, 300)
    # 使用掩码合并边缘和彩色图片
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    # 保存输出图片
    cv2.imwrite(output_path, cartoon)

# 使用函数
cartoonize('storybook/IMG.jpg', 'storybook/output2.jpg')

# import cv2
# import numpy as np

# def cartoonize(image_path, output_path):
#     # Load the image
#     img = cv2.imread(image_path)
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Apply median blur for smoothing
#     gray_blur = cv2.medianBlur(gray, 5)
#     # Use adaptive thresholding to detect edges
#     edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     # Apply bilateral filter to the original image
#     color = cv2.bilateralFilter(img, 9, 300, 300)
#     # Combine edges and colored image using mask
#     cartoon = cv2.bitwise_and(color, color, mask=edges)
#     # Save the output image
#     cv2.imwrite(output_path, cartoon)

# # Use the function
# cartoonize('storybook/IMG-7111.jpg', 'storybook/output2.jpg')
