import numpy as np
import cv2

def add_salt_and_pepper_noise(image, probability):
    output = np.copy(image)
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            if np.random.rand(1) > probability:
                continue
            else:
                if np.random.rand(1) > 0.5:
                    output[x,y] = 0
                else:
                    output[x,y] = 255

    return output.astype(np.uint8)

def copy_padding(image, pad_size):
    height, width = image.shape[:2]
    padded_image = np.zeros((height + 2*pad_size, width + 2*pad_size), dtype=image.dtype)
    # 複製中心
    padded_image[pad_size:height+pad_size, pad_size:width+pad_size] = image
    # 複製上邊
    padded_image[:pad_size, pad_size:width+pad_size] = image[0, :]
    # 複製下邊
    padded_image[height+pad_size:, pad_size:width+pad_size] = image[height-1, :]
    # 複製左邊
    padded_image[pad_size:height+pad_size, :pad_size] = image[:, 0].reshape(height, 1)
    # 複製右邊
    padded_image[pad_size:height+pad_size, width+pad_size:] = image[:, width-1].reshape(height, 1)
    # 複製角落
    padded_image[:pad_size, :pad_size] = image[0, 0]
    padded_image[:pad_size, width+pad_size:] = image[0, width-1]
    padded_image[height+pad_size:, :pad_size] = image[height-1, 0]
    padded_image[height+pad_size:, width+pad_size:] = image[height-1, width-1]

    return padded_image

def mean_filter(image, kernel_size):
    height, width = image.shape[:2]
    border = kernel_size // 2
    padding_img = copy_padding(image,border)
    filtered_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            neighborhood = padding_img[(y+border) - border:(y+border) + border + 1, (x+border) - border:(x+border) + border + 1]
            # 移除 0$255
            valid_pixels = neighborhood[np.where((neighborhood != 0) & (neighborhood != 255))]
            if len(valid_pixels) > 0:
                mean_value = np.mean(valid_pixels)
            else:
                mean_value = 0
            filtered_image[y, x] = mean_value
    

    return filtered_image.astype(np.uint8)