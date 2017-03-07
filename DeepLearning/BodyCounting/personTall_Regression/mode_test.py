# -*-coding=utf-8-*-

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn import svm
from sklearn.externals import joblib
import cv2

x = (
    (375, 261),
    (227, 206),
    (250, 205),
    (330, 256),
    (449, 130),
    (63, 219),
    (96, 148),
    (510, 239),
)
corection = (1, 2.5)
reg = joblib.load('LinearRegression.m')
result = reg.predict(x)
print result
img = cv2.imread('/home/yangzheng/testData/pet/View_001/frame_0000.jpg')
for i in xrange(result.shape[0]):
    startPos = (x[i][0] - (int(result[i, 0]/2 + corection[0])), x[i][1] - (int(result[i, 1]/2 + corection[1])))
    endPos = (x[i][0] + (int(result[i, 0]/2 + corection[0])), x[i][1] + (int(result[i, 1]/2 + corection[1])))
    cv2.rectangle(img, startPos, endPos, color=(0, 0, 255))
cv2.imshow('img', img)
cv2.waitKey(0)
