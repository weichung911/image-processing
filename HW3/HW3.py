from function import *
import cv2 
import numpy as np

# 1.a
baboon_img = cv2.imread("baboon.bmp",0)
baboon_10 = add_salt_and_pepper_noise(baboon_img,0.1)
baboon_30 = add_salt_and_pepper_noise(baboon_img,0.3)
baboon_50 = add_salt_and_pepper_noise(baboon_img,0.5)
baboon_70 = add_salt_and_pepper_noise(baboon_img,0.7)
baboon_90 = add_salt_and_pepper_noise(baboon_img,0.9)
peppers_img = cv2.imread("peppers.bmp",0)
peppers_10 = add_salt_and_pepper_noise(peppers_img,0.1)
peppers_30 = add_salt_and_pepper_noise(peppers_img,0.3)
peppers_50 = add_salt_and_pepper_noise(peppers_img,0.5)
peppers_70 = add_salt_and_pepper_noise(peppers_img,0.7)
peppers_90 = add_salt_and_pepper_noise(peppers_img,0.9)
# cv2.imshow("baboon 10%",baboon_10)
# cv2.imshow("baboon 30%",baboon_30)
# cv2.imshow("baboon 50%",baboon_50)
# cv2.imshow("baboon 70%",baboon_70)
# cv2.imshow("baboon 90%",baboon_90)
# cv2.imshow("peppers 10%",peppers_10)
# cv2.imshow("peppers 30%",peppers_30)
# cv2.imshow("peppers 50%",peppers_50)
# cv2.imshow("peppers 70%",peppers_70)
# cv2.imshow("peppers 90%",peppers_90)
# 1.b
print("mean filtering")
print("before denoise")
print("baboon 10%",cv2.PSNR(baboon_10,baboon_img))
print("baboon 30%",cv2.PSNR(baboon_30,baboon_img))
print("baboon 50%",cv2.PSNR(baboon_50,baboon_img))
print("baboon 70%",cv2.PSNR(baboon_70,baboon_img))
print("baboon 90%",cv2.PSNR(baboon_90,baboon_img))
print("peppers 10%",cv2.PSNR(peppers_10,peppers_img))
print("peppers 30%",cv2.PSNR(peppers_30,peppers_img))
print("peppers 50%",cv2.PSNR(peppers_50,peppers_img))
print("peppers 70%",cv2.PSNR(peppers_70,peppers_img))
print("peppers 90%",cv2.PSNR(peppers_90,peppers_img))
print("after denosise")
baboon_10_mean = mean_filter(baboon_10,5)
baboon_30_mean = mean_filter(baboon_30,5)
baboon_50_mean = mean_filter(baboon_50,5)
baboon_70_mean = mean_filter(baboon_70,5)
baboon_90_mean = mean_filter(baboon_90,5)
peppers_10_mean = mean_filter(peppers_10,5)
peppers_30_mean = mean_filter(peppers_30,5)
peppers_50_mean = mean_filter(peppers_50,5)
peppers_70_mean = mean_filter(peppers_70,5)
peppers_90_mean = mean_filter(peppers_90,5)
print("baboon 10%",cv2.PSNR(baboon_10_mean,baboon_img))
print("baboon 30%",cv2.PSNR(baboon_30_mean,baboon_img))
print("baboon 50%",cv2.PSNR(baboon_50_mean,baboon_img))
print("baboon 70%",cv2.PSNR(baboon_70_mean,baboon_img))
print("baboon 90%",cv2.PSNR(baboon_90_mean,baboon_img))
print("peppers 10%",cv2.PSNR(peppers_10_mean,peppers_img))
print("peppers 30%",cv2.PSNR(peppers_30_mean,peppers_img))
print("peppers 50%",cv2.PSNR(peppers_50_mean,peppers_img))
print("peppers 70%",cv2.PSNR(peppers_70_mean,peppers_img))
print("peppers 90%",cv2.PSNR(peppers_90_mean,peppers_img))
# 1.c

# cv2.waitKey(0)