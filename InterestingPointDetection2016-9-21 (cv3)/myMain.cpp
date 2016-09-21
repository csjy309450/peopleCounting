#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "InterestingPointDetector.h"

using namespace std;
using namespace cv;

/** Function Headers */
void processVideo(const char* videoFilename);

/**
 * Displays instructions on how to use this program.
 */
void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use ViBe with OpenCV                            " << endl
    << "Usage:"                                                                     << endl
    << "./main-opencv <video filename>"                                             << endl
    << "for example: ./main-opencv video.avi"                                       << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}

int main(int argc, char* argv[])
{
    /* Print help information. */
    help();


//    /* Check for the input parameter correctness. */
//    if (argc != 2) {
//        cerr <<"Incorrect input" << endl;
//        cerr <<"exiting..." << endl;
//        return EXIT_FAILURE;
//    }

    /* Create GUI windows. */
    namedWindow("Frame");
    namedWindow("Segmentation by ViBe");

    processVideo("/home/yangzheng/myProgram/test/test.mp4");

    /* Destroy GUI windows. */
    destroyAllWindows();
    return EXIT_SUCCESS;
}

void processVideo(const char* videoFilename)
{
  /* Create the capture object. */
    VideoCapture capture(videoFilename);

    if (!capture.isOpened()) {
        /* Error in opening the video input. */
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }

  /* Variables. */
    static int frameNumber = 1; /* The current frame number */
    Mat frame;                  /* Current frame. */
    Mat outputFrame;
    int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */
    InterestingPointDetector ipd;

  /* Read input data. ESC or 'q' for quitting. */
    while ((char)keyboard != 'q' && (char)keyboard != 27) {
        /* Read the current frame. */
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        // segmentation
        ipd.ProcessFrame(&frame, &outputFrame, frameNumber);

        /* Shows the current frame and the segmentation map. */
        imshow("Frame", frame);
        imshow("Segmentation by ViBe", outputFrame);

        ++frameNumber;

        /* Gets the input from the keyboard. */
        keyboard = waitKey(10);
    }

    /* Delete capture object. */
    capture.release();
}
