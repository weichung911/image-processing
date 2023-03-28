import cv2 as cv
import numpy as np
import math 

def rotate(img,theta=int):
    theta = math.radians(theta)
    height, width, channel = img.shape
    new_height = math.ceil((width*math.sin(theta))+(height*math.cos(theta)))
    new_width = math.ceil((width*math.cos(theta))+(height*math.sin(theta)))
    new_img = np.zeros((new_height,new_width,channel))
    for x in range(width):
        for y in range(height):
            x_ = int((x*math.cos(theta)-y*math.sin(theta))-(width*math.cos(theta)))
            y_ = int(x*math.sin(theta)+y*math.cos(theta))
            if x_ <= new_width & y_ <= new_height:
                new_img[y_,x_] = img[y,x]
    new_img = new_img.astype(np.uint8)
    
    return new_img

def bilinear_resize(img, new_height=int, new_width=int):
    height, width, channel = img.shape
    new_img = np.zeros((new_height,new_width,channel))
    h_scale_factor = height/new_height if new_height != 0 else 0
    w_scale_factor = width/new_width if new_width != 0 else 0
    for h in range(new_height):
        for w in range(new_width):
            x = w * w_scale_factor
            y = h * h_scale_factor

            x_floor = math.floor(x)
            x_ceil = min(width-1,math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(height-1,math.ceil(y))
            if (y_ceil == y_floor) & (x_ceil == x_floor):
                q = img[int(y),int(x), :]
            elif(y_ceil == y_floor):
                q1 = img[int(y),int(x_ceil), :]
                q2 = img[int(y),int(x_floor), :]
                q = (q1 * (x_ceil - x )) + (q2 * (x - x_floor))
            elif(x_ceil == x_floor):
                q1 = img[int(y_ceil),int(x), :]
                q2 = img[int(y_floor),int(x), :]
            else:
                v1 = img[y_floor,x_floor, :]
                v2 = img[y_floor,x_ceil, :]
                v3 = img[y_ceil,x_floor, :]
                v4 = img[y_ceil,x_ceil, :]
                q1 = (v1 * (x_ceil -x)) + (v2 * (x - x_floor))
                q2 = (v3 * (x_ceil - x)) + (v4 * (x - x_floor))
                q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
            new_img[h,w,:] = q
    new_img = new_img.astype(np.uint8)
    return new_img

def overlay(img1,img2):
    height_1, width_1,= img1.shape[0], img1.shape[1]
    for h in range(height_1):
        for w in range(width_1):
            if (img1[h,w,:] != [255,255,255]).all():
                img2[h,w,:] = img1[h,w,:]
    
    img2 = img2.astype(np.uint8)
    return img2