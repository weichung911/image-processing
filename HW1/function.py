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
    for h in range(1,new_height-1):
        for w in range(1,new_width-1):
            if (new_img[h,w] == [0,0,0]).all():
                v1 = new_img[h-1,w]
                v2 = new_img[h+1,w]
                v3 = new_img[h,w+1]
                v4 = new_img[h,w-1]
                new_img[h,w,:] = (v1+v2+v3+v4)/4

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

def intto8binary(n=int):
    return'{0:08b}'.format(n)
def binary2int(n=bytes):
    return int(n,2)
def li_reverse(li):
    new_li = []
    while (li != []):
        new_li.append(li.pop())
    return new_li
def LSB_encode(img1,img2):
    height, width, channel = img2.shape
    img1_h, img1_w = img1.shape[0], img1.shape[1]
    new_img = img2.copy()
    for c in range(channel):
        li = []
        li.append(intto8binary(img1_h))
        li.append(intto8binary(img1_w))
        for h in range(img1_h):
            for w in range(img1_w):
                li.append(intto8binary(img1[h,w,c]))
        li.append('@')
        li = li_reverse(li)
        b_tmp = li.pop()
        counter = 0
        for h in range(height):
            for w in range(width):
                if b_tmp == '@':
                    break 
                if counter != 7:
                    tmp = intto8binary(new_img[h,w,c])
                    tmp = tmp[:7] + b_tmp[counter]
                    new_img[h,w,c] = binary2int(tmp)
                    counter += 1
                else:
                    tmp = intto8binary(new_img[h,w,c])
                    tmp = tmp[:7] + b_tmp[counter]
                    new_img[h,w,c] = binary2int(tmp)
                    counter = 0
                    b_tmp = li.pop()
            if b_tmp == '@':
                break
    return new_img

def LSB_Decode(img):
    height, width, channel = img.shape
    watermark = []
    for c in range(channel):
        li = []
        watermark_h = 0
        watermark_w = 0
        break_flag = False
        b_tmp = ''
        for h in range(height):
            for w in range(width):
                if len(b_tmp) != 7:
                    tmp = intto8binary(img[h,w,c])
                    b_tmp = b_tmp + tmp[7]
                else:
                    tmp = intto8binary(img[h,w,c])
                    b_tmp = b_tmp + tmp[7]
                    if watermark_h == 0:
                        watermark_h = binary2int(b_tmp)
                    elif watermark_w == 0:
                        watermark_w = binary2int(b_tmp)
                    elif (watermark_h*watermark_w) > (len(li)):
                        li.append(binary2int(b_tmp))
                    else:
                        break_flag = True
                    b_tmp = ''
                if break_flag:
                    break
            if break_flag:
                break
        watermark.append(np.array(li).reshape(watermark_h,watermark_w))
    watermark_img = cv.merge((watermark[0], watermark[1], watermark[2]))
    watermark_img = watermark_img.astype(np.uint8)
    return watermark_img


