#include "MatControler.h"

MatControler::MatControler()
{

}

 MatControler::~MatControler()
 {

 }

void MatControler::MatFull(cv::Mat* inputFrame, float x)
 {
    int nRows, nCols, nChannels;
    nRows = inputFrame->rows;
    nCols = inputFrame->cols;
    nChannels = inputFrame->channels();

    if(nChannels==1)
    {
        for(int i=0;i<nRows;i++)
            for(int j=0;j<nCols;j++)
            {
                inputFrame->at<float>(i,j) = x;
            }
    }
    else if(nChannels==3)
    {
        for(int i=0;i<nRows;i++)
            for(int j=0;j<nCols;j++)
            {
                inputFrame->at<float>(i,nChannels*j) = x;
                inputFrame->at<float>(i,nChannels*j+1) = x;
                inputFrame->at<float>(i,nChannels*j+2) = x;
            }
    }

 }

void MatControler::GradientX(cv::Mat* inputFrame, cv::Mat* outputFrame)
{
//    if(nChannels==1)
//    {
//        for(int i=0;i<nRows;i++)
//            for(int j=1;j<nCols-1;j++)
//            {
//                outputFrame->at<float>(i,j) = inputFrame->at<float>(i,j-1) - inputFrame->at<float>(i,j+1);
//            }
//    }
//    else if(nChannels==3)
//    {
//        for(int i=0;i<nRows;i++)
//            for(int j=1;j<nCols-1;j++)
//            {
//                outputFrame->at<float>(i,nChannels*j) = inputFrame->at<float>(i,nChannels*(j-1)) - inputFrame->at<float>(i,nChannels*(j+1));
//                outputFrame->at<float>(i,nChannels*j+1) = inputFrame->at<float>(i,nChannels*(j-1)+1) - inputFrame->at<float>(i,nChannels*(j+1)+1);
//                outputFrame->at<float>(i,nChannels*j+2) = inputFrame->at<float>(i,nChannels*(j-1)+2) - inputFrame->at<float>(i,nChannels*(j+1)+2);
//            }
//    }

    int nRows, nCols, nChannels, dataType;
    nRows = inputFrame->rows;
    nCols = inputFrame->cols;
    nChannels = inputFrame->channels();
    dataType = inputFrame->type();

    if(nChannels == 1)
    {
        if(dataType==CV_8UC1)
        {
            for(int i=0;i<nRows;i++)
                for(int j=1;j<nCols-1;j++)
                {
                    outputFrame->at<float>(i,j) = (float)inputFrame->at<uchar>(i,j-1) - (float)inputFrame->at<uchar>(i,j+1);
                }
        }
        else if(dataType==CV_32FC1)
        {
                for(int i=0;i<nRows;i++)
                    for(int j=1;j<nCols-1;j++)
                    {
                        outputFrame->at<float>(i,j) = inputFrame->at<float>(i,j-1) - inputFrame->at<float>(i,j+1);
                    }
        }
    }
    else if(nChannels == 3)
    {
        if(dataType==CV_8UC3)
        {
            for(int i=0;i<nRows;i++)
                for(int j=1;j<nCols-1;j++)
                {
                    outputFrame->at<float>(i,nChannels*j) = (float)inputFrame->at<uchar>(i,nChannels*(j-1)) - (float)inputFrame->at<uchar>(i,nChannels*(j+1));
                    outputFrame->at<float>(i,nChannels*j+1) = (float)inputFrame->at<uchar>(i,nChannels*(j-1)+1) - (float)inputFrame->at<uchar>(i,nChannels*(j+1)+1);
                    outputFrame->at<float>(i,nChannels*j+2) = (float)inputFrame->at<uchar>(i,nChannels*(j-1)+2) - (float)inputFrame->at<uchar>(i,nChannels*(j+1)+2);
                }
        }
        else if(dataType==CV_32FC3)
        {
            for(int i=0;i<nRows;i++)
                for(int j=1;j<nCols-1;j++)
                {
                    outputFrame->at<float>(i,nChannels*j) = inputFrame->at<float>(i,nChannels*(j-1)) - inputFrame->at<float>(i,nChannels*(j+1));
                    outputFrame->at<float>(i,nChannels*j+1) = inputFrame->at<float>(i,nChannels*(j-1)+1) - inputFrame->at<float>(i,nChannels*(j+1)+1);
                    outputFrame->at<float>(i,nChannels*j+2) = inputFrame->at<float>(i,nChannels*(j-1)+2) - inputFrame->at<float>(i,nChannels*(j+1)+2);
                }
        }
    }
}

void MatControler::GradientY(cv::Mat* inputFrame, cv::Mat* outputFrame)
{
    int nRows, nCols, nChannels, dataType;
    nRows = inputFrame->rows;
    nCols = inputFrame->cols;
    nChannels = inputFrame->channels();
    dataType = inputFrame->type();

    if(nChannels == 1)
    {
        if(dataType==CV_8UC1)
        {
             for(int i=0;i<nCols;i++)
                for(int j=1;j<nRows-1;j++)
                {
                    outputFrame->at<float>(j,i) = inputFrame->at<uchar>(j-1,i) - inputFrame->at<uchar>(j+1,i);
                }
        }
    else if(dataType==CV_32FC1)
    {
             for(int i=0;i<nCols;i++)
                for(int j=1;j<nRows-1;j++)
                {
                    outputFrame->at<float>(j,i) = inputFrame->at<float>(j-1,i) - inputFrame->at<float>(j+1,i);
                }
    }
    }
    else if(nChannels == 3)
    {
        if(dataType==CV_8UC3)
        {
            for(int i=0;i<nCols;i++)
                for(int j=1;j<nRows-1;j++)
                {
                    outputFrame->at<float>(j,nChannels*i) = (float)inputFrame->at<uchar>(j-1,nChannels*i) - (float)inputFrame->at<uchar>(j+1,nChannels*i);
                    outputFrame->at<float>(j,nChannels*i+1) = (float)inputFrame->at<uchar>(j-1,nChannels*i+1) - (float)inputFrame->at<uchar>(j+1,nChannels*i+1);
                    outputFrame->at<float>(j,nChannels*i+2) = (float)inputFrame->at<uchar>(j-1,nChannels*i+2) - (float)inputFrame->at<uchar>(j+1,nChannels*i+2);
                }
        }
        else if(dataType==CV_32FC3)
        {
            for(int i=0;i<nCols;i++)
                for(int j=1;j<nRows-1;j++)
                {
                    outputFrame->at<float>(j,nChannels*i) = inputFrame->at<float>(j-1,nChannels*i) - inputFrame->at<float>(j+1,nChannels*i);
                    outputFrame->at<float>(j,nChannels*i+1) = inputFrame->at<float>(j-1,nChannels*i+1) - inputFrame->at<float>(j+1,nChannels*i+1);
                    outputFrame->at<float>(j,nChannels*i+2) = inputFrame->at<float>(j-1,nChannels*i+2) - inputFrame->at<float>(j+1,nChannels*i+2);
                }
        }
    }
}

void MatControler::MatCotangent(cv::Mat* dividentFrame, cv::Mat* divisorFrame, cv::Mat* outputFrame)
{
    int nRows, nCols, nChannel, dataType;
    nRows = dividentFrame->rows;
    nCols = dividentFrame->cols;
    nChannel = dividentFrame->channels();
    dataType = dividentFrame->type();

    if(nChannel==1)
    {
        if(dataType==CV_32FC1)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
//                    outputFrame->at<float>(i,j) = 1 / ::tan(dividentFrame->at<float>(i,j) / divisorFrame->at<float>(i,j));
                    outputFrame->at<float>(i,j) = cotangent(dividentFrame->at<float>(i,j), divisorFrame->at<float>(i,j));
                }
        }
        else if(dataType==CV_8UC1)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
//                    outputFrame->at<float>(i,j) = 1 / ::tan((float)dividentFrame->at<uchar>(i,j) / (float)divisorFrame->at<uchar>(i,j));
                    outputFrame->at<float>(i,j) = cotangent((float)dividentFrame->at<uchar>(i,j), (float)divisorFrame->at<uchar>(i,j));
                }
        }
    }
    else if(nChannel==3)
    {
        if(dataType==CV_32FC3)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
//                    outputFrame->at<float>(i,nChannel*j) = 1 / ::tan(dividentFrame->at<float>(i,nChannel*j) / divisorFrame->at<float>(i,nChannel*j));
//                    outputFrame->at<float>(i,nChannel*j+1) = 1 / ::tan(dividentFrame->at<float>(i,nChannel*j+1) / divisorFrame->at<float>(i,nChannel*j+1));
//                    outputFrame->at<float>(i,nChannel*j+2) = 1 / ::tan(dividentFrame->at<float>(i,nChannel*j+2) / divisorFrame->at<float>(i,nChannel*j+2));
                    outputFrame->at<float>(i,nChannel*j) = cotangent(dividentFrame->at<float>(i,nChannel*j), divisorFrame->at<float>(i,nChannel*j));
                    outputFrame->at<float>(i,nChannel*j+1) = cotangent(dividentFrame->at<float>(i,nChannel*j+1), divisorFrame->at<float>(i,nChannel*j+1));
                    outputFrame->at<float>(i,nChannel*j+2) = cotangent(dividentFrame->at<float>(i,nChannel*j+2), divisorFrame->at<float>(i,nChannel*j+2));
                }
        }
        else if(dataType==CV_8UC3)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
//                    outputFrame->at<float>(i,nChannel*j) = 1 / ::tan((float)dividentFrame->at<uchar>(i,nChannel*j) / (float)divisorFrame->at<uchar>(i,nChannel*j));
//                    outputFrame->at<float>(i,nChannel*j+1) = 1 / ::tan((float)dividentFrame->at<uchar>(i,nChannel*j+1) / (float)divisorFrame->at<uchar>(i,nChannel*j+1));
//                    outputFrame->at<float>(i,nChannel*j+2) = 1 / ::tan((float)dividentFrame->at<uchar>(i,nChannel*j+2) / (float)divisorFrame->at<uchar>(i,nChannel*j+2));
                    outputFrame->at<float>(i,nChannel*j)  = cotangent((float)dividentFrame->at<uchar>(i,nChannel*j), (float)divisorFrame->at<uchar>(i,nChannel*j));
                    outputFrame->at<float>(i,nChannel*j+1)  = cotangent((float)dividentFrame->at<uchar>(i,nChannel*j+1), (float)divisorFrame->at<uchar>(i,nChannel*j+1));
                    outputFrame->at<float>(i,nChannel*j+2)  = cotangent((float)dividentFrame->at<uchar>(i,nChannel*j+2), (float)divisorFrame->at<uchar>(i,nChannel*j+2));
                }
        }
    }
}

void MatControler::EuclideanNorm(cv::Mat* FrameX, cv::Mat* FrameY, cv::Mat* outputFrame)
{
    int nRows, nCols, nChannel, dataType;
    nRows = FrameX->rows;
    nCols = FrameX->cols;
    nChannel = FrameX->channels();
    dataType = FrameX->type();

    if(nChannel==1)
    {
        if(dataType==CV_8UC1)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
                    outputFrame->at<float>(i,j) = ::sqrt(::pow((float)FrameX->at<uchar>(i,j), 2) + ::pow((float)FrameY->at<uchar>(i,j), 2));
                }
        }
        else if(dataType==CV_32FC1)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
                    outputFrame->at<float>(i,j) = ::sqrt(::pow(FrameX->at<float>(i,j), 2) + ::pow(FrameY->at<float>(i,j), 2));
                }
        }
    }
    else if(nChannel==3)
    {
        if(dataType==CV_8UC3)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
                    outputFrame->at<float>(i,nChannel*j) = ::sqrt(::pow((float)FrameX->at<uchar>(i,nChannel*j), 2) + ::pow((float)FrameY->at<uchar>(i,nChannel*j), 2));
                    outputFrame->at<float>(i,nChannel*j+1) = ::sqrt(::pow((float)FrameX->at<uchar>(i,nChannel*j+1), 2) + ::pow((float)FrameY->at<uchar>(i,nChannel*j+1), 2));
                    outputFrame->at<float>(i,nChannel*j+2) = ::sqrt(::pow((float)FrameX->at<uchar>(i,nChannel*j+2), 2) + ::pow((float)FrameY->at<uchar>(i,nChannel*j+2), 2));
                }
        }
        else if(dataType==CV_32FC3)
        {
            for(int i=0;i<nRows;i++)
                for(int j=0;j<nCols;j++)
                {
                    outputFrame->at<float>(i,nChannel*j) = ::sqrt(::pow(FrameX->at<float>(i,nChannel*j), 2) + ::pow(FrameY->at<float>(i,nChannel*j), 2));
                    outputFrame->at<float>(i,nChannel*j+1) = ::sqrt(::pow(FrameX->at<float>(i,nChannel*j+1), 2) + ::pow(FrameY->at<float>(i,nChannel*j+1), 2));
                    outputFrame->at<float>(i,nChannel*j+2) = ::sqrt(::pow(FrameX->at<float>(i,nChannel*j+2), 2) + ::pow(FrameY->at<float>(i,nChannel*j+2), 2));
                }
        }
    }
}

void Uniformization(cv::Mat* Frame)
{

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// non-member functions

float cotangent(float divident, float divisor) //1/tan(divident/divisor)
{
    if(divident==0)
    {
        return 1000;
    }
    if(divisor==0)
    {
        return 1/::tan(divident/1);
    }
    return 1/::tan(divident/divisor);
}
