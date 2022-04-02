# -*- coding:utf-8 -*-
# @Time : 2022/2/24 0:29
# @Author : 西~南~北
# @File : Split_ch.py
# @Software: PyCharm

import cv2
from matplotlib import pyplot as plt


class Split_ch:
    def __init__(self, img):
        self.img = img  # 读取图片
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化

        # 图像阈值化操作——获得二值化图
        self.ret, image = cv2.threshold(self.img_gray, 0, 255, cv2.THRESH_OTSU)
        # image=cv2.bitwise_not(image)

        # 形态学操作
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image = cv2.dilate(image, self.kernel)
        # 查找轮廓
        self.contours, self.hierarchy = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.words = []
        self.word_images = []

    def fun(self):
        for item in self.contours:
            word = []
            rect = cv2.boundingRect(item)
            x = rect[0]
            y = rect[1]
            weight = rect[2]
            height = rect[3]
            word.append(x)
            word.append(y)
            word.append(weight)
            word.append(height)
            self.words.append(word)
        words = sorted(self.words, key=lambda s: s[0], reverse=False)
        i = 0
        for word in words:
            # 根据轮廓的外接矩形筛选轮廓
            if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10):
                i = i + 1
                splite_image = self.image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
                splite_image = cv2.resize(splite_image, (25, 40))
                self.word_images.append(splite_image)
        fig = plt.figure()
        for i, j in enumerate(self.word_images):
            plt.subplot(1, 7, i + 1)
            plt.imshow(self.word_images[i], cmap='gray')

        plt.show()

# split = Split_ch('license_plate/one/liaoA09030.jpg')
# split.fun()
