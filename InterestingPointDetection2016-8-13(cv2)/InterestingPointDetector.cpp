#include "InterestingPointDetector.h"

InterestingPointDetector::InterestingPointDetector()
{
}

InterestingPointDetector::~InterestingPointDetector()
{
    g_x.release();
    g_y.release();
    mat_O.release();
    mat_M.release();
}

bool InterestingPointDetector::ProcessFrame(cv::Mat* inputFrame, cv::Mat* outputFrame, long frameNum)
{
    if(frameNum==1)
    {
        if(inputFrame->channels()==1)
        {
            g_y = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC1);
            g_x = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC1);
            mat_O = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC1);
            mat_M = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC1);
        }
        else if(inputFrame->channels()==3)
        {
            g_y = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC3);
            g_x = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC3);
            mat_O = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC3);
            mat_M = cv::Mat(inputFrame->rows, inputFrame->cols, CV_32FC3);
        }
    }
    //initial the working mats
    m_matControler.MatFull(&g_x, 255.0);
    m_matControler.MatFull(&g_y, 255.0);
    m_matControler.MatFull(&mat_O, 255.0);
    m_matControler.MatFull(&mat_M, 255.0);

    m_matControler.GradientX(inputFrame, &g_x);
    m_matControler.GradientY(inputFrame, &g_y);
    m_matControler.MatCotangent(&g_y, &g_x, &mat_O);
    m_matControler.EuclideanNorm(&g_x, &g_y, &mat_M);

    *outputFrame = mat_M.clone();

//    //test code
//    std::cout<<inputFrame<<std::endl;
//    std::cout<<g_x<<std::endl;
//    std::cout<<g_y<<std::endl;
//    std::cout<<mat_O<<std::endl;
//    std::cout<<mat_M<<std::endl;

    return true;
}
