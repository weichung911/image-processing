import math 
from  function import rotate, LSB_encode, LSB_Decode
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

# print(math.cos(math.radians(15)))


def intto8bits(n=int):
    return'{0:08b}'.format(n)
def binary2int(n=bytes):
    return int(n,2)

# a = intto8bits(50)
# print(a)
# b = '1'
# a = a[:7] + b
# print(a[8])


graveler = cv.imread('graveler.bmp')
lena = cv.imread('lena.bmp')


img = LSB_encode(graveler,lena)
# cv.imshow('img',img)
img2 = LSB_Decode(img)
cv.imshow('img2',img2)
cv.waitKey(0)







