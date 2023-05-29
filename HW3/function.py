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

def adaptive_median_filter(image):
    height, width = image.shape[:2]
    filter_img = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            kernel_size = 1
            valid_pixels = []
            while len(valid_pixels)==0:
                # print(kernel_size)
                border = kernel_size // 2
                if y-border>=0 and y+border<=height and x-border>=0 and x+border<=width:
                    neighborhood = image[y - border:y + border + 1, x - border:x + border + 1]
                    valid_pixels = neighborhood[np.where((neighborhood != 0) & (neighborhood != 255))]
                    if len(valid_pixels) > 0:
                        filter_img[y,x] = round(np.median(valid_pixels))
                        break
                    else:
                        kernel_size += 2
                else:
                    filter_img[y,x] = 0
                    break
    return filter_img.astype(np.uint8)

def sobel_edge_detection(image):
    # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    # blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Define the Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Convolve the image with the Sobel kernels
    gradient_x = cv2.filter2D(image, -1, sobel_x)
    gradient_y = cv2.filter2D(image, -1, sobel_y)

    # Calculate the magnitude and direction of gradients
    # magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = abs(gradient_x) + abs(gradient_y)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))  # Convert back to uint8

    # Normalize the magnitude to [0, 255]
    # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold the magnitude to obtain the edge map
    threshold_value = 127 # Adjust this value to control the edge detection sensitivity
    _, edge_map = cv2.threshold(magnitude, threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return edge_map