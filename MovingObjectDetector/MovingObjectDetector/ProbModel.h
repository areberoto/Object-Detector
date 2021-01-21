#pragma once

#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

class ProbModel{
private:
	//parameters
	int blockSize;
	int blockSizeSqr;
	float varianceInterpolateParam;

	float maxBGAge;
	float varMinNoiseT;
	float varDecRatio;
	float minBGVar;
	float initBGVar;

	int numModels;
	float varThreshFGDetermine;
	float varThreshModelMatch;

	//variables
	Mat frameGray;
	Mat mDistImg;

	float* mMeanM1, * mMeanM2;
	float* mVarM1, * mVarM2;
	float* mAgeM1, * mAgeM2;


	float* mMeanM1Tmp, * mMeanM2Tmp;
	float* mVarM1Tmp, * mVarM2Tmp;
	float* mAgeM1Tmp, * mAgeM2Tmp;

	int* mModelID;
	
	int modelWidth;
	int modelHeight;
	int obsWidth;
	int obsHeight;

public:
	ProbModel();
	~ProbModel();
	void init(Mat& imgGray);
	void motionCompensate(Mat H);
	void update(Mat& outputImg);
};

