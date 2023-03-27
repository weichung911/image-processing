import cv2 as cv
import numpy as np
import math 

def rotate(img,theta=int):
    theta = math.radians(theta)
    height, width, channel = img.shape
    new_height = math.ceil((width*math.sin(theta))+(height*math.cos(theta)))
    new_width = math.ceil((width*math.cos(theta))+(height*math.sin(theta)))
    print(height,new_height)
    print(width,new_width)
    new_img = np.zeros((new_height,new_width,channel))
    for x in range(width):
        for y in range(height):
            x_ = int((x*math.cos(theta)-y*math.sin(theta))-(width*math.cos(theta)))
            y_ = int(x*math.sin(theta)+y*math.cos(theta))
            if x_ <= new_width & y_ <= new_height:
                new_img[y_,x_] = img[y,x]
    new_img = new_img.astype(np.uint8)
    
    return new_img