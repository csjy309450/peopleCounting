# -*-encoding=utf-8-*-
import copy
import os
import os.path as path
import cv2
import datetime

import BodyDetector.body as body
import BodyDetector.BodyDetection as bd
import BackgroudSegmenatation as bs

param_dict_ucsd = {
    "imags_directory": "",
    "input_shape": [1, 24, 24, 3],
    "person_tall_model_path": "./personTall_Regression/ucsd_personTall_Model.m",
    "person_wh_correction": (0.5, 1.5),
    "bouding_box_correction": (3, 6),
    "nms_thresh": 0.1,
    "backgroud_seg_mode": bs.Mode_BackgroudSegmentation['Mode_KNN'],
}

param_dict_pet = {
    "imags_directory": "",
    "input_shape": [1, 24, 24, 3],
    "person_tall_model_path": "./personTall_Regression/pet_personTall_Model.m",
    "person_wh_correction": (2, 4),
    "bouding_box_correction": (4, 8),
    "nms_thresh": 0.3,
    "backgroud_seg_mode": bs.Mode_BackgroudSegmentation['Mode_KNN'],
}

param_dict_pet = {
    "imags_directory": "",
    "input_shape": [1, 24, 24, 3],
    "person_tall_model_path": "./personTall_Regression/mall_personTall_Model.m",
    "person_wh_correction": (0.5, 1.5),
    "bouding_box_correction": (3, 6),
    "nms_thresh": 0.3,
    "backgroud_seg_mode": bs.Mode_BackgroudSegmentation['Mode_KNN'],
}

imags_directory_list = [
    "/home/yangzheng/testData/mall_dataset/frames/Sample-213x160"
]
result_dir_path = "./result_images/mall_result"
param_dict = param_dict_pet

for im_dir in imags_directory_list:
    param_dict["imags_directory"] = im_dir
    cur_result_dir = path.join(result_dir_path, path.split(im_dir)[1])
    try:
        os.makedirs(cur_result_dir)
    except Exception, e:
        if e.args[1] != "File exists":
            exit(0)

    mybd = bd.BodyDetection(param_dict)
    person_count_file = path.join(cur_result_dir, "person_count.txt")
    f = open(person_count_file, mode="w")
    while mybd.mybs.isnot_take_out():
        frameStartTime = datetime.datetime.now()
        mybd.mybs.ImgSeqProcessing()
        frame = mybd.mybs.getCurrentFrame()
        boundingRect = mybd.mybs.getBoundingRect()
        frame_file_name = path.splitext(mybd.mybs.frame_file_name())

        if len(boundingRect) <= 2:
            continue

        # ## display background segmentation result
        # cv2.imshow('background segmentation', mybd.mybs.get_bgs_image())
        # cv2.imshow('background mask', mybd.mybs.fgmask)
        # cv2.imshow('contours mask', mybd.mybs.contoursMask)
        cv2.imwrite(path.join(cur_result_dir, frame_file_name[0] + "_bs" + frame_file_name[1]),
                    mybd.mybs.get_bgs_image())
        cv2.imwrite(path.join(cur_result_dir, frame_file_name[0] + "_cm" + frame_file_name[1]), mybd.mybs.contoursMask)

        print "*************************************"
        mybd.detect_full_image(frame, boundingRect)
        ## output result
        PerosnArray = mybd.getRealPersonArray()
        print "personCount is: ", PerosnArray.shape[0]
        f.write(str(PerosnArray.shape[0]) + "\n")

        frameEndTime = datetime.datetime.now()
        print "time is: ", (frameEndTime - frameStartTime).seconds, "(s)", \
            (frameEndTime - frameStartTime), "(μs)"

        ## display result image
        reulte = copy.deepcopy(frame)
        for it in PerosnArray:
            cv2.rectangle(reulte, (int(it[0]), int(it[1])),
                          (int(it[2]), int(it[3])),
                          (0, 0, 255))

        # cv2.imshow('org img', frame)
        # cv2.imshow('reulte', reulte)
        cv2.imwrite(path.join(cur_result_dir, frame_file_name[0] + "_result" + frame_file_name[1]), reulte)
        # cv2.waitKey(0)

    f.close()
