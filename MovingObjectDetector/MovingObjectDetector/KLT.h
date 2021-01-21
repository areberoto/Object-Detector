#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/calib3d.hpp"
#include <vector>
#include <iostream>

using namespace cv;
using std::vector;

class KLT{
private:
	Mat img, H, maskImg, frameGrayPrev, swapTemp;
	vector<Point2f> p0, p1;
	vector<uchar> status;
	int winSize, count, flags;

	void swapData(Mat imgGray);
	void makeHomography(vector<int> nMatch);

public:
	KLT();
	void init(const Mat& imgGray);
	void initFeatures(Mat imgGray);
	void runTrack(Mat imgGray);
	Mat getHomography();
};

