#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "MC.h"

using std::cerr;

int main(){
    MC *mMC = new MC();

    VideoCapture cap("chita.mp4");
    if (!cap.isOpened()) {
        cerr << "Unable to open video file!\n";
        return 0;
    }

    namedWindow("Output video", WINDOW_AUTOSIZE);

    int frameNum{ 1 };
    bool run{ true };
    Mat frameCopy, rawImg;

    while (run) {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        resize(frame, frame, Size(320, 240));

        frameCopy = frame.clone();
        rawImg = frame.clone();

        if (frameNum == 1)
            mMC->init(rawImg);
        else
            mMC->run();

        float drawOrig{ 0.5 };
        for (int i = 0; i < frameCopy.rows; i++){
            for (int j = 0; j < frameCopy.cols; j++){
                frameCopy.at<Vec3b>(i, j) = drawOrig * frameCopy.at<Vec3b>(i, j);

                if (frameNum > 1) {
                    Vec3b bgrPixel = frameCopy.at<Vec3b>(i, j);
                    bgrPixel[2] += frameCopy.at<uchar>(i, j) > 0 ? 255 * (1.0 - drawOrig) : 0;
                    frameCopy.at<Vec3b>(i, j) = bgrPixel;
                }
            }
        }

        //draw results as overlay
            //for (int j{ 0 }; j < frameCopy.rows; j++) {
            //    for (int i{ 0 }; i < frameCopy.cols; i++) {
            //        

            //        unsigned char* pMaskImg = (unsigned char*)(mMC->detectImg.data);
            //        //int widthStepMsk = mMC->detectImg.step;

            //        int widthTmp = mMC->detectImg.cols;
            //        int nchannelsTmp = mMC->detectImg.channels();

            //        int widthStepMsk = ((widthTmp * sizeof(unsigned char) * nchannelsTmp) % 4 != 0) ? ((((widthTmp * sizeof(unsigned char) * nchannelsTmp) / 4) * 4) + 4) : (widthTmp * sizeof(unsigned char) * nchannelsTmp);
            //        //int widthStepMsk = 320 * 4;

            //        int maskData = pMaskImg[i + j * widthStepMsk];

            //        ((unsigned char*)(frameCopy.data))[i * 3 + j * frameCopy.step + 2] = drawOrig * ((unsigned char*)(frameCopy.data))[i * 3 + j * frameCopy.step + 2];
            //        ((unsigned char*)(frameCopy.data))[i * 3 + j * frameCopy.step + 1] = drawOrig * ((unsigned char*)(frameCopy.data))[i * 3 + j * frameCopy.step + 1];
            //        ((unsigned char*)(frameCopy.data))[i * 3 + j * frameCopy.step + 0] = drawOrig * ((unsigned char*)(frameCopy.data))[i * 3 + j * frameCopy.step + 0];

            //        if (frameNum > 1)
            //            ((unsigned char*)(frameCopy.data))[i * 3 + j * frameCopy.step + 2] += maskData > 0 ? 255 * (1.0 - drawOrig) : 0;
            //    }
            //}

        

        imshow("Output video", frameCopy);

        switch (waitKey(1)) {
        case 'q':
            run = false;
            break;
        default:
            break;
        }

        frameNum++;
    }

    destroyAllWindows();
    return 0;
}