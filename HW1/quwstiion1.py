import cv2 as cv
import numpy as np 
from  function import rotate

laptop_left = cv.imread('laptop_left.png')
laptop_right = cv.imread('laptop_right.png')


laptop = np.zeros((laptop_left.shape[0],laptop_left.shape[1]+laptop_right.shape[1],3))

for h in range(laptop.shape[0]):
    for w in range(laptop.shape[1]):
        if w < laptop_left.shape[1]:
            laptop[h,w] = laptop_left[h,w]
        if w >= laptop_left.shape[1]:
            laptop[h,w] = laptop_right[h,(w-laptop_left.shape[1])]
laptop = laptop.astype(np.uint8)

cv.imshow('laptop',laptop)

new_img = rotate(laptop,35)
cv.imshow('new_img',new_img)
cv.waitKey(0)