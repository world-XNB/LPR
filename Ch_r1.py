# -*- coding:utf-8 -*-
# @Time : 2022/2/24 0:29
# @Author : 西~南~北
# @File : Split_ch.py
# @Software: PyCharm

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Ch_r1:
    def __init__(self, img):
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
        self.words = []
        self.word_images = []

        # 模版匹配
        # 准备模板(template[0-9]为数字模板；)
        self.template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
                         'V', 'W', 'X', 'Y', 'Z',
                         '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁',
                         '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']
        self.chinese_words_list = self.get_chinese_words_list()
        self.eng_words_list = self.get_eng_words_list()
        self.eng_num_words_list = self.get_eng_num_words_list()
        self.result = None

    def Split(self):
        # 图像阈值化操作——获得二值化图
        ret, image = cv2.threshold(self.img_gray, 0, 255, cv2.THRESH_OTSU)
        # image=cv2.bitwise_not(image)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.dilate(image, kernel)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for item in contours:
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
        self.words = sorted(self.words, key=lambda s: s[0], reverse=False)
        i = 0
        for word in self.words:
            # 根据轮廓的外接矩形筛选轮廓
            if (word[3] > (word[2] * 1)) and (word[3] < (word[2] * 3)) and (word[2] > 10):
                i = i + 1
                splite_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
                splite_image = cv2.resize(splite_image, (25, 40))
                self.word_images.append(splite_image)

        fig = plt.figure()
        for i, j in enumerate(self.word_images):
            plt.subplot(1, 7, i + 1)
            plt.imshow(self.word_images[i], cmap='gray')
        plt.savefig('./2.jpg')
        # plt.show()

    # 读取一个文件夹下的所有图片，输入参数是文件名，返回模板文件地址列表
    def read_directory(self, directory_name):
        referImg_list = []
        for filename in os.listdir(directory_name):
            referImg_list.append(directory_name + "/" + filename)
        return referImg_list

    # 获得中文模板列表（只匹配车牌的第一个字符）
    def get_chinese_words_list(self):
        chinese_words_list = []
        for i in range(34, 64):
            # 将模板存放在字典中
            c_word = self.read_directory('./refer1/' + self.template[i])
            chinese_words_list.append(c_word)
        return chinese_words_list

    # 获得英文模板列表（只匹配车牌的第二个字符）
    def get_eng_words_list(self):
        eng_words_list = []
        for i in range(10, 34):
            e_word = self.read_directory('./refer1/' + self.template[i])
            eng_words_list.append(e_word)
        return eng_words_list

    # 获得英文和数字模板列表（匹配车牌后面的字符）
    def get_eng_num_words_list(self):
        eng_num_words_list = []
        for i in range(0, 34):
            word = self.read_directory('./refer1/' + self.template[i])
            eng_num_words_list.append(word)
        return eng_num_words_list

    # 读取一个模板地址与图片进行匹配，返回得分
    def template_score(self, template, image):
        # 将模板进行格式转换
        template_img = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
        # 模板图像阈值化处理——获得黑白图
        ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
        #     height, width = template_img.shape
        #     image_ = image.copy()
        #     image_ = cv2.resize(image_, (width, height))
        image_ = image.copy()
        # 获得待检测图片的尺寸
        height, width = image_.shape
        # 将模板resize至与图像一样大小
        template_img = cv2.resize(template_img, (width, height))
        # 模板匹配，返回匹配得分
        result = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
        return result[0][0]

    # 对分割得到的字符逐一匹配
    def template_matching(self, word_images):
        results = []
        for index, word_image in enumerate(word_images):
            if index == 0:
                best_score = []
                for chinese_words in self.chinese_words_list:
                    score = []
                    for chinese_word in chinese_words:
                        result = self.template_score(chinese_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                # print(template[34+i])
                r = self.template[34 + i]
                results.append(r)
                continue
            if index == 1:
                best_score = []
                for eng_word_list in self.eng_words_list:
                    score = []
                    for eng_word in eng_word_list:
                        result = self.template_score(eng_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                # print(template[10+i])
                r = self.template[10 + i]
                results.append(r)
                continue
            else:
                best_score = []
                for eng_num_word_list in self.eng_num_words_list:
                    score = []
                    for eng_num_word in eng_num_word_list:
                        result = self.template_score(eng_num_word, word_image)
                        score.append(result)
                    best_score.append(max(score))
                i = best_score.index(max(best_score))
                # print(template[i])
                r = self.template[i]
                results.append(r)
                continue
        return results

    def fun(self):
        self.Split()
        word_images_ = self.word_images.copy()
        # 调用函数获得结果
        self.result = self.template_matching(word_images_)
        print(self.result)
        # "".join(self.result)  # 函数将列表转换为拼接好的字符串，方便结果显示
        # print("".join(self.result))

# img = cv2.imread('license_plate/one/liaoA09030.jpg')  # 读取图片
# ch = Ch_r1(img)
# ch.fun()
