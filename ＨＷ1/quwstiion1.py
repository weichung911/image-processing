import cv2 as cv
import numpy as np 

laptop_left = cv.imread('laptop_left.png')
laptop_right = cv.imread('laptop_right.png')
print(laptop_left.shape[0])
print(laptop_right.shape)

laptop = np.zeros((laptop_left.shape[0],laptop_left.shape[1]+laptop_right.shape[1],3))

for h in range(laptop.shape[0]):
    for w in range(laptop.shape[1]):
        if w < laptop_left.shape[1]:
            laptop[h,w] = laptop_left[h,w]
        if w >= laptop_left.shape[1]:
            laptop[h,w] = laptop_right[h,(w-laptop_left.shape[1])]
laptop = laptop.astype(np.uint8)
print(laptop[0,281])
print(laptop_right[0,0])
cv.imshow('laptop',laptop)
cv.waitKey(0)