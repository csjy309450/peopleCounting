/**
*Copyright(C), 2016/8, YangZheng
*FileName: InterestingPointDetector.h
*Author: YangZheng
*Date: 2016/8/13
*Description: None
**/
#ifndef INTERESTINGPOINTDETECTOR_H_INCLUDED
#define INTERESTINGPOINTDETECTOR_H_INCLUDED

#include <iostream>
#include <opencv/cv.h>
#include "MatControler.h"

using namespace std;
using namespace cv;

using namespace cv;

class InterestingPointDetector
{
public:
    InterestingPointDetector();
    ~InterestingPointDetector();
    bool ProcessFrame(cv::Mat* inputFrame, cv::Mat* outputFrame, long frameNum);
private:
    MatControler m_matControler;
    cv::Mat g_x;
    cv::Mat g_y;
    cv::Mat mat_O;
    cv::Mat mat_M;
};


#endif // INTERESTINGPOINTDETECTOR_H_INCLUDED
