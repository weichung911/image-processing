import cv2 as cv
import numpy as np 
from  function import rotate, bilinear_resize, overlay

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

lena = cv.imread('lena.bmp')
print(lena.shape)
cv.imshow('original_lena',lena)
resize_lena = bilinear_resize(lena,1024,1024)
print(resize_lena.shape)
cv.imshow('resize_lena',resize_lena)

graveler = cv.imread('graveler.bmp')
graveler_lena = overlay(graveler,lena)
cv.imshow('gravelr_lena',graveler_lena)
cv.waitKey(0)