import cv2 as cv
import numpy as np 
from  function import rotate, bilinear_resize, overlay, LSB_encode, LSB_Decode, LSB_Decode_jpeg, PSNR 
from PIL import Image
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
cv.imshow('rotate_laptop',new_img)
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
lena = cv.flip(lena, 1)

img = LSB_encode(graveler,lena)
cv.imshow('lena_watermark',img)
img2 = LSB_Decode(img)
cv.imshow('extract_watermark',img2)
cv.waitKey(0)
# Q5 (b)
for i in range(3):
    ratios = [1,50,100]
    # 压缩为JPEG格式
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), ratios[i]] # JPEG压缩质量为70%
    result, imgencode = cv.imencode('.jpg', img, encode_param)

    # 将压缩后的数据解码为图像数据
    imgdecode = cv.imdecode(imgencode, cv.IMREAD_COLOR)

    # 将解码后的图像数据保存到文件
    cv.imwrite('output'+str(i)+'.jpg', imgdecode)

output0 = cv.imread('output0.jpg')
de_output0 = LSB_Decode_jpeg(output0)
output1 = cv.imread('output1.jpg')
de_output1 = LSB_Decode_jpeg(output1)
output2 = cv.imread('output2.jpg')
de_output2 = LSB_Decode_jpeg(output2)
cv.imshow('de0',de_output0)
print(PSNR(graveler,de_output0))
cv.imshow('de1',de_output1)
print(PSNR(graveler,de_output1))
cv.imshow('de2',de_output2)
print(PSNR(graveler,de_output2))
cv.waitKey(0)