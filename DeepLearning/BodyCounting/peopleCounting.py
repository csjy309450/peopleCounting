# -*-encoding=utf-8-*-
import copy
import cv2
import datetime


import body.BodyDetection as bd
import BackgroudSegmenatation as bs

param_dict_ucsd = {
    "imags_directory": "/home/yangzheng/testData/ucsd/vidf1_33_000.y",
    "input_shape": [1, 24, 24, 3],
    "person_tall_model_path": "./personTall_Regression/ucsd_personTall_Model.m",
    "person_wh_correction": (0.5, 1.5),
    "bouding_box_correction": (3, 6),
    "nms_thresh": 0.1,
    "backgroud_seg_mode": bs.Mode_BackgroudSegmentation['Mode_KNN'],
}

param_dict_pet = {
    "imags_directory": "/home/yangzheng/testData/pet/View_001",
    "input_shape": [1, 24, 24, 3],
    "person_tall_model_path": "./personTall_Regression/pet_personTall_Model.m",
    "person_wh_correction": (2, 4),
    "bouding_box_correction": (4, 8),
    "nms_thresh": 0.3,
    "backgroud_seg_mode": bs.Mode_BackgroudSegmentation['Mode_KNN'],
}

mybd = bd.BodyDetection(param_dict_ucsd)

while 1:
    frameStartTime = datetime.datetime.now()
    mybd.mybs.ImgSeqProcessing()
    frame = mybd.mybs.getCurrentFrame()
    boundingRectList = mybd.mybs.getBoundingRect()

    ##test
    frame_copy = copy.deepcopy(frame)
    frame_copy = mybd.mybs.DrawBoudningBox(frame_copy, boundingRectList)
    cv2.imshow('background segmentation', frame_copy)
    cv2.imshow('background mask', mybd.mybs.fgmask)

    if len(boundingRectList) <= 2:
        continue
    print "*************************************"
    mybd.detect_full_image(frame, boundingRectList)
    reulte = copy.deepcopy(frame)
    PerosnArray = mybd.getRealPersonArray()
    for it in PerosnArray:
        cv2.rectangle(reulte, (int(it[0]), int(it[1])),
                      (int(it[2]), int(it[3])),
                      (0, 0, 255))

    print "personCount is: ", PerosnArray.shape[0]
    cv2.imshow('org img', frame)
    cv2.imshow('reulte', reulte)

    frameEndTime = datetime.datetime.now()
    print "time is: ", (frameEndTime-frameStartTime).seconds

    cv2.waitKey()