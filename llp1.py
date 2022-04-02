# -*- coding:utf-8 -*-
# @Time : 2022/2/24 0:19
# @Author : 西~南~北
# @File : llp1.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import cv2


class llp:
    def __init__(self, fname):
        # 读取待检测图片
        self.img = cv2.imread(fname)
        self.reimg = None

    # plt显示彩色图片
    def plt_show0(self, img):
        # cv2与plt的图像通道不同：cv2为[b,g,r];plt为[r, g, b]
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        cv2.imwrite("1.jpg", img)
        # plt.imshow(img)
        # plt.show()

    # plt显示灰度图片
    def plt_show(self, img):
        plt.imshow(img, cmap='gray')
        plt.show()

    # 图像去噪灰度处理
    def gray_guss(self, image):
        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image

    def fun(self):
        # 复制一张图片，在复制图上进行图像操作，保留原图
        image = self.img.copy()
        # 图像去噪灰度处理
        gray_image = self.gray_guss(image)
        # x方向上的边缘检测（增强边缘信息）
        Sobel_x = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0)
        absX = cv2.convertScaleAbs(Sobel_x)
        image = absX

        # 图像阈值化操作——获得二值化图
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

        # 形态学（从图像中提取对表达和描绘区域形状有意义的图像分量）——闭操作
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=1)

        # 腐蚀（erode）和膨胀（dilate）
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        # x方向进行闭操作（抑制暗细节）
        image = cv2.dilate(image, kernelX)
        image = cv2.erode(image, kernelX)
        # y方向的开操作
        image = cv2.erode(image, kernelY)
        image = cv2.dilate(image, kernelY)
        # 中值滤波（去噪）
        image = cv2.medianBlur(image, 21)
        # 获得轮廓
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for item in contours:
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            # 根据轮廓的形状特点，确定车牌的轮廓位置并截取图像
            if (weight > (height * 2.5)) and (weight < (height * 4)):
                self.reimg = self.img[y:y + height, x:x + weight]
                self.plt_show0(self.reimg)

# ll = llp('./license_plate/three/jingC88888.jpg')
# ll.fun()
