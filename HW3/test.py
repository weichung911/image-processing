import numpy as np
import cv2
from function import *

def copy_padding(image, pad_size):
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 创建新的补齐图像
    padded_image = np.zeros((height + 2*pad_size, width + 2*pad_size), dtype=image.dtype)

    # 复制图像中心区域
    padded_image[pad_size:height+pad_size, pad_size:width+pad_size] = image

    # 复制上边界
    padded_image[:pad_size, pad_size:width+pad_size] = image[0, :]

    # 复制下边界
    padded_image[height+pad_size:, pad_size:width+pad_size] = image[height-1, :]

    # 复制左边界
    padded_image[pad_size:height+pad_size, :pad_size] = image[:, 0].reshape(height, 1)

    # 复制右边界
    padded_image[pad_size:height+pad_size, width+pad_size:] = image[:, width-1].reshape(height, 1)

    # 复制四个角
    padded_image[:pad_size, :pad_size] = image[0, 0]
    padded_image[:pad_size, width+pad_size:] = image[0, width-1]
    padded_image[height+pad_size:, :pad_size] = image[height-1, 0]
    padded_image[height+pad_size:, width+pad_size:] = image[height-1, width-1]

    return padded_image

# 读取图像
image = cv2.imread('baboon.bmp', 0)  # 以灰度模式读取图像

# 设置补齐大小
pad_size = 10

# 进行复制填充
padded_image = copy_padding(image, pad_size)
m = mean_filter(image,5)
# 显示图像
cv2.imshow('Padded Image', padded_image)
cv2.imshow('m',m)
cv2.waitKey(0)
cv2.destroyAllWindows()
