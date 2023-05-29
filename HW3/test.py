
from function import *
import numpy as np
import scipy.ndimage
import cv2

# # 读取图像
image = cv2.imread('peppers.bmp', 0)  # 以灰度模式读取图像

if image is None:
    print("Failed to load image")
    exit()
img = sobel_edge_detection(image)
laplacian_noisy = cv2.Laplacian(img, cv2.CV_64F)
cv2.imshow("image",image)
cv2.imshow("img",img)
cv2.imshow("lp",laplacian_noisy)
# cv2.waitKey(0)
def Sobel_edge_detection(f):
    grad_x = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize = 3)
    grad_y = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize = 3)
    magnitude = abs(grad_x) + abs(grad_y)
    g = np.uint8(np.clip(magnitude, 0, 255))
    ret, g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return g
d = Sobel_edge_detection(image)
cv2.imshow('d',d)
cv2.waitKey(0)
# # 定义高斯滤波器的参数
# size = 5  # 滤波器大小
# sigma = 2  # 标准差

# # 创建高斯滤波器
# x, y = np.meshgrid(np.linspace(-size // 2, size // 2, size), np.linspace(-size // 2, size // 2, size))
# kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
# kernel /= np.sum(kernel)

# # 应用高斯滤波器
# filtered_image = scipy.ndimage.convolve(image, kernel)
# cv2.imshow('Original Image', image)
# cv2.imshow('Filtered Image', filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(5//2)
# a = [1,2,3]
# id =1 

# while len(a)>0:
#     print(a)
#     a =[]