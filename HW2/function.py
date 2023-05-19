import cv2
import numpy as np
import matplotlib.pyplot as plt

def closing(image, kernel_size, dilation_iterations, erosion_iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated_image = cv2.dilate(image, kernel, iterations=dilation_iterations)
    closed_image = cv2.erode(dilated_image, kernel, iterations=erosion_iterations)
    return closed_image

def linear_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = (255 / (max_val - min_val)) * (image - min_val)
    return stretched_image.astype(np.uint8)

def histogram_equalization(image):
    # 計算灰度直方圖
    hist = [0] * 256
    total_pixels = image.shape[0] * image.shape[1]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            hist[pixel_value] += 1
    # 計算CDF
    cdf = [0] * 256
    cdf[0] = hist[0] / total_pixels

    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i] / total_pixels
    
    # 將CDF映射到新的灰度值範圍
    cdf_normalized = [int(round(value * 255)) for value in cdf]

    # 將灰度值映射到新的範圍
    equalized_image = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            equalized_image[i, j] = cdf_normalized[pixel_value]

    return equalized_image

def two_histogram_equalization(image):
    # 計算灰度直方圖
    hist = [0] * 256
    total_pixels = image.shape[0] * image.shape[1]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            hist[pixel_value] += 1
            
    # μ 
    median = 0
    cumulative_sum = 0
    for i in range(256):
        cumulative_sum += hist[i]
        if cumulative_sum >= total_pixels // 2:
            median = i
            break
    # two sub-histograms
    hist_lower = hist[:median+1]
    hist_upper = hist[median+1:]

    # 計算CDF
    cdf_lower = [0] * len(hist_lower)
    cdf_upper = [0] * len(hist_upper)
    cdf_lower[0] = hist_lower[0] / sum(hist_lower)
    cdf_upper[0] = hist_upper[0] / sum(hist_upper)

    for i in range(1, len(hist_lower)):
        cdf_lower[i] = cdf_lower[i-1] + hist_lower[i] / sum(hist_lower)

    for i in range(1, len(hist_upper)):
        cdf_upper[i] = cdf_upper[i-1] + hist_upper[i] / sum(hist_upper)

    
    # 將CDF映射到新的灰度值範圍
    lut_lower = [round(value * median) for value in cdf_lower]
    lut_upper = [round(value * (255 - median) + median + 1) for value in cdf_upper]

    # 將灰度值映射到新的範圍
    equalized_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            if pixel_value <= median:
                equalized_image[i, j] = lut_lower[pixel_value]
            else:
                equalized_image[i, j] = lut_upper[pixel_value - median - 1]

    return equalized_image

def enhance_contrast(image):
    # 定義窗口大小和半窗口大小
    window_size = 7
    half_window = window_size // 2

    # 創建與原始圖像相同大小的輸出圖像
    output_image = np.zeros_like(image)

    # 迭代遍歷每個像素
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 提取局部窗口
            window = image[max(i - half_window, 0):min(i + half_window + 1, image.shape[0]),
                        max(j - half_window, 0):min(j + half_window + 1, image.shape[1])]
            
            # 計算局部窗口的直方圖
            hist = [0] * 256
            for pixel_value in window.flatten():
                hist[pixel_value] += 1
            
            # 計算局部窗口的累積直方圖
            cdf = [0] * 256
            cdf[0] = hist[0]
            for k in range(1, 256):
                cdf[k] = cdf[k-1] + hist[k]
            
            # 計算增強函數
            enhancement_func = int((cdf[int(image[i, j])] * 255) / cdf[-1])
            
            # 將增強後的像素值賦值給輸出圖像
            output_image[i, j] = enhancement_func

    return output_image.astype(np.uint8)