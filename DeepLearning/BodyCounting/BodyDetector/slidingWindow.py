#! /usr/bin/python2.7
# -*-encoding=utf-8-*-

import os.path as path
import cv2
from DocManager import DocManager

DirPath = '/home/yangzheng/testData/ucsd/vidf1_33_000.y'
DocM = DocManager.DocManager(_docFilter=['.png'])
DocList = DocM.GetDocuments(DirPath)
print DocList

width = 15
height = 30
center_offset_w = width/2
center_offset_h = height/2

for it in DocList:
    img = cv2.imread(path.join(DirPath, it))
    w_indx = img.shape[1]-width/2
    h_indx = img.shape[0]-height/2
    print w_indx, h_indx
    for h in xrange(h_indx):
        for w in xrange(w_indx):
            # sub_img = cv2.getRectSubPix(img, (width, height), (w+center_offset_w, h+center_offset_h))
            # cv2.imshow('sub img', sub_img)
            cv2.rectangle(img, (w, h), (w + width, h + height), (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(0)



