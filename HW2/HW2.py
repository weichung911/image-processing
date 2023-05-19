import cv2 as cv 
from function import *
# 1.1
text_img = cv.imread("text-broken.tif")
kernel_size = (3, 3)
fix_text_img = closing(text_img,kernel_size,2,2)
cv.imshow('fix_text',fix_text_img)
cv.imwrite('fix_text.tif',fix_text_img)
# 1.2
edges = cv2.Canny(fix_text_img, 100, 200)
cv.imshow('boundaries',edges)
cv.imwrite('boundaries.tif',edges)
# 2 
einstein_img = cv.imread("einstein-low-contrast.tif")
stretch_einstein_img = linear_stretching(einstein_img)
cv.imshow('stretch',stretch_einstein_img)
cv2.imwrite('streth.tif',stretch_einstein_img)
# 3
aerialview_img = cv.imread("aerialview-washedout.tif", 0)
HE_aerialview_img = histogram_equalization(aerialview_img)
cv.imshow('HE',HE_aerialview_img)
cv.imwrite('HE.tif',HE_aerialview_img)
# 4
two_HE = two_histogram_equalization(aerialview_img)
cv.imshow('two_HE',two_HE)

# 5
enhance_einstein_img = enhance_contrast(einstein_img)
cv.imshow('enhance_einstein_img',enhance_einstein_img)
cv.waitKey(0)
