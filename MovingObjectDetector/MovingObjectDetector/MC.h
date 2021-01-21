#pragma once

#include <opencv2/highgui.hpp>
#include "KLT.h"
#include "ProbModel.h"

using namespace cv;

class MC {
private:
	Mat frame, frameGray, frameGrayPrev, H;
	KLT lucasKanade;
	ProbModel BGmodel;

public:
	Mat detectImg;
	MC();
	void init(const Mat& src);
	void run();
};

