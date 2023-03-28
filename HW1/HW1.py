import cv2 as cv
import numpy as np 
from  function import rotate, bilinear_resize, overlay

# Q1
laptop_left = cv.imread('laptop_left.png')
laptop_right = cv.imread('laptop_right.png')

laptop = np.zeros((laptop_left.shape[0],laptop_left.shape[1]+laptop_right.shape[1],3))

laptop[:,:laptop_left.shape[1],:] = laptop_left
laptop[:,laptop_left.shape[1]:,:] = laptop_right
laptop = laptop.astype(np.uint8)

cv.imshow('laptop',laptop)
# Q2
new_img = rotate(laptop,15)
cv.imshow('new_img',new_img)
# Q3
lena = cv.imread('lena.bmp')
print(lena.shape)
cv.imshow('original_lena',lena)
resize_lena = bilinear_resize(lena,1024,1024)
print(resize_lena.shape)
cv.imshow('resize_lena',resize_lena)
# Q4
graveler = cv.imread('graveler.bmp')
graveler_lena = overlay(graveler,lena)
cv.imshow('gravelr_lena',graveler_lena)
cv.waitKey(0)