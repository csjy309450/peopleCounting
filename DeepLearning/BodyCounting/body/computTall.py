# -*-coding=utf-8-*-

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn import svm
from sklearn.externals import joblib
import cv2

x = (
    (133, 123),
    (80, 128),
    (92, 77),
    (196, 49),
    (13, 122)
)
corection = (1, 2.5)
reg = joblib.load('/home/yangzheng/myPrograms/personsCounting/personTall_Regression/personTall_Model.m')
result = reg.predict(x)
print result
img = cv2.imread('/home/yangzheng/testData/ucsd/vidf1_33_000.y/vidf1_33_000_f012.png')
for i in xrange(result.shape[0]):
    startPos = (x[i][0] - (int(result[i, 0]/2 + corection[0])), x[i][1] - (int(result[i, 1]/2 + corection[1])))
    endPos = (x[i][0] + (int(result[i, 0]/2 + corection[0])), x[i][1] + (int(result[i, 1]/2 + corection[1])))
    cv2.rectangle(img, startPos, endPos, color=(0, 0, 255))
cv2.imshow('img', img)
cv2.waitKey(0)
