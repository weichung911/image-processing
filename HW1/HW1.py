import cv2 as cv
import numpy as np 
from  function import rotate, bilinear_resize, overlay, LSB_encode, LSB_Decode 

# Q1
laptop_left = cv.imread('laptop_left.png')
laptop_right = cv.imread('laptop_right.png')

laptop = np.zeros((laptop_left.shape[0],laptop_left.shape[1]+laptop_right.shape[1],3))

laptop[:,:laptop_left.shape[1],:] = laptop_left
laptop[:,laptop_left.shape[1]:,:] = laptop_right
laptop = laptop.astype(np.uint8)

cv.imshow('laptop',laptop)
cv.waitKey(0)
# Q2
new_img = rotate(laptop,15)
cv.imshow('new_img',new_img)
cv.waitKey(0)
# Q3
lena = cv.imread('lena.bmp')
print(lena.shape)
cv.imshow('original_lena',lena)
resize_lena = bilinear_resize(lena,1024,1024)
print(resize_lena.shape)
cv.imshow('resize_lena',resize_lena)
cv.waitKey(0)
# Q4
graveler = cv.imread('graveler.bmp')
graveler_lena = overlay(graveler,lena)
cv.imshow('gravelr_lena',graveler_lena)
cv.waitKey(0)
# Q5 (a)
graveler = cv.imread('graveler.bmp')
lena = cv.imread('lena.bmp')

img = LSB_encode(graveler,lena)
img2 = LSB_Decode(img)
cv.imshow('img2',img2)
cv.waitKey(0)
# Q5 (b)
