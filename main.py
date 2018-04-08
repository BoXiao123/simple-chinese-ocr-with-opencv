#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import cv2
import numpy as np
from train import classify
#得到透视变换矩阵
def warp(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
    return warped, Minv

#透视变换
def perspective_image(img,
                      src,
                      dst= np.float32([
                            [0,0],
                            [600,0],
                            [0,1200],
                            [600,1200]]),):

    img, Minv = warp(img, src, dst)
    return img, Minv

#膨胀和腐蚀
def preprocessimage(img,n):
    kernel=np.uint8(np.zeros((n,n)))
    for x in range(n):
        kernel[x,2]=1
        kernel[2,x]=1
    eroded=cv2.erode(img,kernel)
    dilated = cv2.dilate(img, kernel)
    result = cv2.absdiff(dilated, eroded)
    return result

#主函数
if __name__=='__main__':
    #参数
    dir_name='test_imgs'
    img_names=os.listdir(dir_name)
    img_names.sort()
    src=[np.float32([[202,265],[889,289],[43,1187],[992,1219]]),
         np.float32([[202,265],[889,289],[43,1187],[992,1219]]),
         np.float32([[202,265],[889,289],[43,1187],[992,1219]]),
         np.float32([[0,0],[600,0],[0,800],[600,800],]),
         np.float32([[0,0],[600,0],[0,800],[600,800],])]
    #循环读入图像
    for f,img_name in enumerate(img_names):
        img_path=os.path.join(dir_name,img_name)
        #loop all images
        img=cv2.imread(img_path)
        #得到透视变换的ROI
        perspective, Minv = perspective_image(img,src[f])
        perspective=perspective[0:1200,0:600]
        #bgr转hsv,抠出黑色部分
        hsv = cv2.cvtColor(perspective, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0,0,0])
        upper_black = np.array([200,255,54])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        res = cv2.bitwise_and(perspective,perspective, mask= mask)
        #膨胀腐蚀
        processed=preprocessimage(res,20)
        gray = cv2.cvtColor(processed,cv2.COLOR_BGR2GRAY)
        #二值化
        ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        #检测轮廓
        _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        x, y, w, h = cv2.boundingRect(largest_contour)
        #cv2.rectangle(perspective, (x,y), (x+w, y+h), (0, 255, 0), 2)
        croped=perspective[y:y+h,x:x+w]
        text='crop%d.jpg'%f
        _, binary = cv2.threshold(croped, 100, 255, cv2.THRESH_BINARY)
        #字符分割
        kernel = np.uint8(np.zeros((3, 3)))
        for x in range(3):
            kernel[x, 2] = 1
            kernel[2, x] = 1
        eroded = cv2.erode(binary, kernel)
        binary_img = eroded[:, :, 0]
        js = []
        w = binary_img.shape[1]
        h = binary_img.shape[0]
        for x in range(w):
            j = 0
            for y in range(h):
                if binary_img[y][x] < 10:
                    j += 1
            js.append(j)

        positions = []
        for position, j in enumerate(js):
            # print j
            if position < len(js) - 1:
                if j == 0 and js[position + 1] != 0:
                    positions.append(position)
        results='The recognition reult is: '
        positions.append(w)
        if not os.path.exists('croped%d'%f):
            os.mkdir('croped%d'%f)
        for i in range(len(positions) - 1):
            seg = binary[:, positions[i]:positions[i + 1]]
            cv2.imwrite('croped%d/%d.jpg'%(f,i),seg)
            #深度学习做分类
            results+=classify('croped%d/%d.jpg'%(f,i))
        #打印结果
        print results


