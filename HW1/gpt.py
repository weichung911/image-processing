import numpy as np

# Load grayscale image
gray_image = np.array([[120, 50, 200], [30, 80, 100], [180, 130, 70]])

# Convert each pixel value to 8-bit binary
binary_image = np.array([list(bin(pixel)[2:].zfill(8)) for pixel in gray_image.flatten()], dtype=int)

# Reshape binary image back to original shape
binary_image = binary_image.reshape(gray_image.shape[0], gray_image.shape[1], 8)

# Convert binary image to 8 separate layers
layers = [binary_image[:, :, i] for i in range(8)]

print(layers)