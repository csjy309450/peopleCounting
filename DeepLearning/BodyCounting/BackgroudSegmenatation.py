# -*-encoding=utf-8-*-

import os.path as path
import numpy as np
import copy
import cv2
import DocManager.DocManager as dm

# imgDir = '/home/yangzheng/testData/ucsd/vidf1_33_000.y'
# docM = dm.DocManager(_docFilter=['.png'])
# imgsList = docM.GetFileList(imgDir)
# print imgsList
Mode_BackgroudSegmentation = {
    'Mode_KNN': 0,
}

class BackgroudSegmentation:
    def __init__(self, param_dict, person_tall_reg):
        self.param_dict = param_dict
        self.imgDir = self.param_dict["imags_directory"]
        self.docM = dm.DocManager(_docFilter=['.jpg', '.png'])
        self.imgsList = self.docM.GetFileList(self.imgDir)
        self.nframe = 0
        self.person_tall_reg = person_tall_reg
        if self.param_dict["backgroud_seg_mode"] == Mode_BackgroudSegmentation['Mode_KNN']:
            self.fgbg = cv2.createBackgroundSubtractorKNN()
        else:
            pass

    def Dilate(self, img, dilate_elem=0, dilate_size=3):
        """
        膨胀运算
        :param img:
        :param dilate_elem:
        :param dilate_size:
        :return:
        """
        if dilate_elem == 0:
            dilate_type = cv2.MORPH_RECT
        elif dilate_elem == 1:
            dilate_type = cv2.MORPH_CROSS
        elif dilate_elem == 2:
            dilate_type = cv2.MORPH_ELLIPSE
        kernel = cv2.getStructuringElement(dilate_type, (dilate_size, dilate_size))
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        return img

    def Erode(self, img, erode_elem=0, erode_size=3):
        """
        腐蚀运算
        :param img:
        :param erode_elem:
        :param erode_size:
        :return:
        """
        if erode_elem == 0:
            erode_type = cv2.MORPH_RECT
        elif erode_elem == 1:
            erode_type = cv2.MORPH_CROSS
        elif erode_elem == 2:
            erode_type = cv2.MORPH_ELLIPSE
        kernel = cv2.getStructuringElement(erode_type, (erode_size, erode_size))
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        return img

    def FindBoudingBox(self, img, corection=(0, 0)):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        _, binary_img = cv2.threshold(gray_img, 127, 255,0)
        temp = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = temp[1]
        boundingRect = []

        ## fill contours
        self.contoursMask = np.zeros_like(self.frame)
        for i in xrange(len(contours)):
            cv2.drawContours(self.contoursMask, contours, i, (255, 255, 255), cv2.FILLED)
        ## display valid points
        # cv2.imshow('contours mask', self.contoursMask)

        for points in contours:
            r = cv2.boundingRect(points)
            center = [(int(r[0] + r[2]/2), int(r[1] + r[3]/2))]
            personSize = self.comput_person_wh(center, corection=self.param_dict["person_wh_correction"])

            if r[2] >= personSize[0]-corection[0] and r[3] >= personSize[1]-corection[1]:
                boundingRect.append(copy.deepcopy(r))
        # print boundingRect
        return boundingRect

    def get_contours_mask(self):
        return self.contoursMask

    def comput_person_wh(self, pos, corection=(0, 0)):
        personRect = self.person_tall_reg.predict(pos)
        personSize = (int(personRect[0, 0]+corection[0]), int(personRect[0, 1]+corection[1]))
        return personSize

    def DrawBoudningBox(self, img, boundingRect):
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        img_ = copy.deepcopy(img)
        for rect in boundingRect:
            startPos = (int(rect[0]), int(rect[1]))
            endPos = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
            img_ = cv2.rectangle(img_, startPos, endPos, color=(0, 0, 255))
        return img_

    def ImgSeqProcessing_test(self):
        for imgPath in self.imgsList:
            self.frame = cv2.imread(path.join(self.imgDir, imgPath))
            self.fgmask = self.fgbg.apply(self.frame)
            self.fgmask = self.Dilate(self.fgmask, 0, 1)
            self.fgmask = self.Dilate(self.fgmask, 0, 2)
            self.fgmask = self.Erode(self.fgmask, 1, 2)
            self.fgmask = self.Erode(self.fgmask, 1, 1)

            self.boundingRect = self.FindBoudingBox(self.fgmask, corection=self.param_dict["bouding_box_correction"])
            self.outputImage = self.DrawBoudningBox(self.frame, self.boundingRect)

            # print gMM2.apply
            cv2.imshow('org img', self.frame)
            cv2.imshow('mask', self.fgmask)
            cv2.imshow('reulte', self.outputImage)
            cv2.waitKey()
        cv2.destroyAllWindows()

    def frame_file_name(self):
        return self.imgsList[self.nframe]

    def ImgSeqProcessing(self):
        """
        读取一帧图像提取前景图像,寻找有效Rect搜索区域
        :return:
        """
        if not self.isnot_take_out():
            return
        ## 加载图像
        self.frame = cv2.imread(path.join(self.imgDir, self.imgsList[self.nframe]))
        if self.frame.shape[0] == 0:
            return
        self.nframe += 1
        ## 前景提取
        self.fgmask = self.fgbg.apply(self.frame)
        ## 形态学滤波
        self.fgmask = self.Dilate(self.fgmask, 0, 1)
        self.fgmask = self.Dilate(self.fgmask, 0, 2)
        # self.fgmask = self.Erode(self.fgmask, 1, 1)
        self.fgmask = self.Erode(self.fgmask, 1, 1)
        ## 找到有效Rect搜索区域
        self.boundingRect = self.FindBoudingBox(self.fgmask, corection=self.param_dict["bouding_box_correction"])

    def get_bgs_image(self):
        bgs_img = self.DrawBoudningBox(self.frame, self.boundingRect)
        return bgs_img

    def getCurrentFrame(self):
        return self.frame

    def getForegroudMask(self):
        return self.fgmask

    def getBoundingRect(self):
        return self.boundingRect

    def getNFrame(self):
        return self.nframe

    def isnot_take_out(self):
        # print not(self.nframe == len(self.imgsList)-1)
        return not (self.nframe == len(self.imgsList)-1)

if __name__ == '__main__':
    imgDir = '/home/yangzheng/testData/pet/test/gammaCorrection'
    test = BackgroudSegmentation(imgDir, _mode=Mode_BackgroudSegmentation['Mode_KNN'])
    while 1:
        test.ImgSeqProcessing()
        cv2.imshow('org img', test.getCurrentFrame())
        cv2.imshow('mask', test.getForegroudMask())
        cv2.imshow('reulte', test.getOutputImage())
        cv2.waitKey()

