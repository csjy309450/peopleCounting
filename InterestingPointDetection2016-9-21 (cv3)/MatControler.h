/*****
* Copyright(C), 2016~, YangZheng
* FileName: MatControler.h
* Author: YangZheng
* Date: 2016/8/13
* Description: None
*****/
#ifndef MATCONTROLER_H_INCLUDED
#define MATCONTROLER_H_INCLUDED

#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

float cotangent(float divident, float divisor);

class MatControler
{
public:
    MatControler();
    ~MatControler();
    /**
    * To full the Mat
    **/
    void MatFull(cv::Mat* inputFrame, float x);
    /**
    * To compute the gradient in direction x
    **/
    void GradientX(cv::Mat* inputFrame, cv::Mat* outputFrame);
    /**
    * To compute the gradient in direction y
    **/
    void GradientY(cv::Mat* inputFrame, cv::Mat* outputFrame);
    /**
    * To compute the elements' cotangent between two Mat
    **/
    void MatCotangent(cv::Mat* dividentFrame, cv::Mat* divisorFrame, cv::Mat* outputFrame);
    /**
    * To compute the matrix norm
    **/
    void EuclideanNorm(cv::Mat* FrameX, cv::Mat* FrameY, cv::Mat* outputFrame);
    /**
    * 将矩阵归一化到0~255区间
    **/
    void Uniformization(cv::Mat* Frame);
};


#endif // MATCONTROLER_H_INCLUDED
