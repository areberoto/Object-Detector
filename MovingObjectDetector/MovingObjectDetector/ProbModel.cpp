#include "ProbModel.h"

ProbModel::ProbModel() : mMeanM1{ nullptr }, mMeanM2{ nullptr },
	mVarM1{ nullptr }, mVarM2{ nullptr }, mAgeM1{ nullptr }, mAgeM2{ nullptr },
	mMeanM1Tmp{ nullptr }, mMeanM2Tmp{ nullptr }, mVarM1Tmp{ nullptr }, mVarM2Tmp{ nullptr },
	mAgeM1Tmp{ nullptr }, mAgeM2Tmp{ nullptr }, mModelID{ nullptr }{
	
	blockSize = 4;
	blockSizeSqr = 16;
	numModels = 2;
	varianceInterpolateParam = 1.0;
	maxBGAge = 30.0;
	varMinNoiseT = 2500.0;
	varDecRatio = 0.001f;
	minBGVar = 25.0;
	initBGVar = 400.0;
	varThreshFGDetermine = 4.0;
	varThreshModelMatch = 2.0;
	modelWidth = 0;
	modelHeight = 0;
	obsHeight = 0;
	obsWidth = 0;
}

ProbModel::~ProbModel() {

	if (NULL != mMeanM1)
		delete[] mMeanM1;
	if (NULL != mMeanM2)
		delete[] mMeanM2;
	if (NULL != mVarM1)
		delete[] mVarM1;
	if (NULL != mVarM2)
		delete[] mVarM2;
	if (NULL != mAgeM1)
		delete[] mAgeM1;
	if (NULL != mAgeM2)
		delete[] mAgeM2;
	
	if (NULL != mMeanM1Tmp)
		delete[] mMeanM1Tmp;
	if (NULL != mMeanM2Tmp)
		delete[] mMeanM2Tmp;
	if (NULL != mVarM1Tmp)
		delete[] mVarM1Tmp;
	if (NULL != mVarM2Tmp)
		delete[] mVarM2Tmp;
	if (NULL != mAgeM1Tmp)
		delete[] mAgeM1Tmp;
	if (NULL != mAgeM2Tmp)
		delete[] mAgeM2Tmp;

	if (NULL != mModelID)
		delete[] mModelID;
}

void ProbModel::init(Mat& imgGray) {
	frameGray = imgGray;

	obsWidth = imgGray.cols;
	obsHeight = imgGray.rows;

	modelWidth = static_cast<int>(obsWidth / blockSize);
	modelHeight = static_cast<int>(obsHeight / blockSize);

	//Chech for ini
	mDistImg = Mat(imgGray.size(), imgGray.type());

	//Allocate memory for main arrays
	mMeanM1 = new float[modelWidth * modelHeight];
	mMeanM2 = new float[modelWidth * modelHeight];
	mVarM1 = new float[modelWidth * modelHeight];
	mVarM2 = new float[modelWidth * modelHeight];
	mAgeM1 = new float[modelWidth * modelHeight];
	mAgeM2 = new float[modelWidth * modelHeight];

	//set all elements of main arrays to 0
	memset(mMeanM1, 0, sizeof(float) * modelWidth * modelHeight);
	memset(mMeanM2, 0, sizeof(float) * modelWidth * modelHeight);
	memset(mVarM1, 0, sizeof(float) * modelWidth * modelHeight);
	memset(mVarM2, 0, sizeof(float) * modelWidth * modelHeight);
	memset(mAgeM1, 0, sizeof(float) * modelWidth * modelHeight);
	memset(mAgeM2, 0, sizeof(float) * modelWidth * modelHeight);

	//Allocate memory for temporal arrays
	mMeanM1Tmp = new float[modelWidth * modelHeight];
	mMeanM2Tmp = new float[modelWidth * modelHeight];
	mVarM1Tmp = new float[modelWidth * modelHeight];
	mVarM2Tmp = new float[modelWidth * modelHeight];
	mAgeM1Tmp = new float[modelWidth * modelHeight];
	mAgeM2Tmp = new float[modelWidth * modelHeight];
	
	mModelID = new int[modelWidth * modelHeight];

	//Set homography identity matrix
	Mat H = (Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); //Identity

	motionCompensate(H);
	//Mat tmp(240, 320, imgGray.type());
	//update(tmp);
}

void ProbModel::motionCompensate(Mat H) {
	//Compensate models for current view
	for (int j{ 0 }; j < modelHeight; j++) {
		for (int i{ 0 }; i < modelWidth; i++) {
			//x and y coordinates for current model
			//center of Grid NxN 
			float X{ static_cast<float>(blockSize * i + blockSize / 2.0) };
			float Y{ static_cast<float>(blockSize * j + blockSize / 2.0) };
			float W{ 1.0 };

			//std::cout << H.type() << std:: endl;

			//Transoform coordinates with H
			float newW = H.at<float>(0, 2) * X + H.at<float>(1, 2) * Y + H.at<float>(2, 2);
			float newX = (H.at<float>(0, 0) * X + H.at<float>(1, 0) * Y + H.at<float>(2, 0)) / newW;
			float newY = (H.at<float>(0, 1) * X + H.at<float>(1, 1) * Y + H.at<float>(2, 1)) / newW;

			//Transformed i, j coordinates of old position
			float newI{ newX / blockSize };
			float newJ{ newY / blockSize };

			int idxNewI{ static_cast<int>(floor(newI)) };
			int idxNewJ{ static_cast<int>(floor(newJ)) };

			float di = static_cast<float>(newI - (idxNewI + 0.5));
			float dj = static_cast<float>(newJ - (idxNewJ + 0.5));
			
			float wHorizontal{ 0.0 };
			float wVertical{ 0.0 };
			float wHV{ 0.0 };
			float wSelf{ 0.0 };
			float sumW{ 0.0 };

			int idxNow{ i + j * modelWidth };

			//FOR MEAN AND AGE
			float tempMean[4][2]; //2 refers to number of models
			float tempAge[4][2];
			memset(tempMean, 0, sizeof(float) * 4 * numModels);
			memset(tempAge, 0, sizeof(float) * 4 * numModels);

			//Horizontal Neighbor
			if (di != 0) {
				int idxNewII{ idxNewI };
				int idxNewJJ{ idxNewJ };

				idxNewII += di > 0 ? 1 : -1;
				if (idxNewII >= 0 && idxNewII < modelWidth && idxNewJJ >= 0 && idxNewJJ < modelHeight) {
					wHorizontal = static_cast<float>( fabs(di) * (1.0 - fabs(dj)) );
					sumW += wHorizontal;
					int idxNew{ idxNewII + idxNewJJ * modelWidth };
					tempMean[0][0] = wHorizontal * mMeanM1[idxNew];
					tempMean[0][1] = wHorizontal * mMeanM2[idxNew];
					tempAge[0][0] = wHorizontal * mAgeM1[idxNew];
					tempAge[0][1] = wHorizontal * mAgeM2[idxNew];
				}
			}

			//Vertical Neighbor
			if (dj != 0) {
				int idxNewII{ idxNewI };
				int idxNewJJ{ idxNewJ };

				idxNewJJ += dj > 0 ? 1 : -1;
				if (idxNewII >= 0 && idxNewII < modelWidth && idxNewJJ >= 0 && idxNewJJ < modelHeight) {
					wVertical = fabs(dj) * (1.0 - fabs(di));
					sumW += wVertical;
					int idxNew{ idxNewII + idxNewJJ * modelWidth };
					tempMean[1][0] = wVertical * mMeanM1[idxNew];
					tempMean[1][1] = wVertical * mMeanM2[idxNew];
					tempAge[1][0] = wVertical * mAgeM1[idxNew];
					tempAge[1][1] = wVertical * mAgeM2[idxNew];
				}
			}

			//Horizontal Vertical neighbor
			if (dj != 0 && di != 0) {
				int idxNewII{ idxNewI };
				int idxNewJJ{ idxNewJ };

				idxNewII += di > 0 ? 1 : -1;
				idxNewJJ += dj > 0 ? 1 : -1;
				if (idxNewII >= 0 && idxNewII < modelWidth && idxNewJJ >= 0 && idxNewJJ < modelHeight) {
					wHV = fabs(di) * fabs(dj);
					sumW += wHV;
					int idxNew{ idxNewII + idxNewJJ * modelWidth };
					tempMean[2][0] = wHV * mMeanM1[idxNew];
					tempMean[2][1] = wHV * mMeanM2[idxNew];
					tempAge[2][0] = wHV * mAgeM1[idxNew];
					tempAge[2][1] = wHV * mAgeM2[idxNew];
				}
			}

			//Self
			if (idxNewI >= 0 && idxNewI < modelWidth && idxNewJ >= 0 && idxNewJ < modelHeight) {
				wSelf = (1.0 - fabs(di)) * (1.0 - fabs(dj));
				sumW += wSelf;

				int idxNew{ idxNewI + idxNewJ * modelWidth };
				tempMean[3][0] = wSelf * mMeanM1[idxNew];
				tempMean[3][1] = wSelf * mMeanM2[idxNew];
				tempAge[3][0] = wSelf * mAgeM1[idxNew];
				tempAge[3][1] = wSelf * mAgeM2[idxNew];
			}

			if (sumW > 0) {
				mMeanM1Tmp[idxNow] = (tempMean[0][0] + tempMean[1][0] + tempMean[2][0] + tempMean[3][0]) / sumW;
				mMeanM2Tmp[idxNow] = (tempMean[0][1] + tempMean[1][1] + tempMean[2][1] + tempMean[3][1]) / sumW;

				mAgeM1Tmp[idxNow] = (tempAge[0][0] + tempAge[1][0] + tempAge[2][0] + tempAge[3][0]) / sumW;
				mAgeM2Tmp[idxNow] = (tempAge[0][1] + tempAge[1][1] + tempAge[2][1] + tempAge[3][1]) / sumW;

				//mMeanM1Tmp[idxNow] = tempMean[3][0] / sumW;
				//mMeanM2Tmp[idxNow] = tempMean[3][1] / sumW;

				//mAgeM1Tmp[idxNow] = tempAge[3][0] / sumW;
				//mAgeM2Tmp[idxNow] = tempAge[3][1] / sumW;
			}

			//FOR VARIANCE
			float tempVar[4][2];
			memset(tempVar, 0, sizeof(float) * 4 * numModels);

			//Horizontal neighbor
			if (di != 0) {
				int idxNewII{ idxNewI };
				int idxNewJJ{ idxNewJ };
				idxNewII += di > 0 ? 1 : -1;
				if (idxNewII >= 0 && idxNewII < modelWidth && idxNewJJ >= 0 && idxNewJJ < modelHeight) {
					int idxNew{ idxNewII + idxNewJJ * modelWidth };
					tempVar[0][0] = wHorizontal * (mVarM1[idxNew] + varianceInterpolateParam * (pow(mMeanM1Tmp[idxNow] - mMeanM1[idxNew], static_cast<int>(2))));
					tempVar[0][1] = wHorizontal * (mVarM2[idxNew] = varianceInterpolateParam * (pow(mMeanM2Tmp[idxNow] - mMeanM2[idxNew], static_cast<int>(2))));
				}
			}

			//Vertical neighbor
			if (dj != 0) {
				int idxNewII{ idxNewI };
				int idxNewJJ{ idxNewJ };

				idxNewJJ += dj > 0 ? 1 : -1;
				if (idxNewII >= 0 && idxNewII < modelWidth && idxNewJJ >= 0 && idxNewJJ < modelHeight) {
					int idxNew{ idxNewII + idxNewJJ * modelWidth };
					tempVar[1][0] = wVertical * (mVarM1[idxNew] + varianceInterpolateParam * (pow(mMeanM1Tmp[idxNow] - mMeanM1[idxNew], static_cast<int>(2))));
					tempVar[1][1] = wVertical * (mVarM2[idxNew] + varianceInterpolateParam * (pow(mMeanM2Tmp[idxNow] - mMeanM2[idxNew], static_cast<int>(2))));
				}
			}

			//Horizontal Vertical neighbor
			if (dj != 0 && di != 0) {
				int idxNewII{ idxNewI };
				int idxNewJJ{ idxNewJ };

				idxNewII += di > 0 ? 1 : -1;
				idxNewJJ += dj > 0 ? 1 : -1;
				if (idxNewII >= 0 && idxNewII < modelWidth && idxNewJJ >= 0 && idxNewJJ < modelHeight) {
					int idxNew{ idxNewII + idxNewJJ * modelWidth };
					tempVar[2][0] = wHV * (mVarM1[idxNew] + varianceInterpolateParam * (pow(mMeanM1Tmp[idxNow] - mMeanM1[idxNew], static_cast<int>(2))));
					tempVar[2][1] = wHV * (mVarM2[idxNew] + varianceInterpolateParam * (pow(mMeanM2Tmp[idxNow] - mMeanM2[idxNew], static_cast<int>(2))));
				}
			}

			//Self
			if (idxNewI >= 0 && idxNewI < modelWidth && idxNewJ >= 0 && idxNewJ < modelHeight) {
				int idxNew{ idxNewI + idxNewJ * modelWidth };
				tempVar[3][0] = wSelf * (mVarM1[idxNew] + varianceInterpolateParam * (pow(mMeanM1Tmp[idxNow] - mMeanM1[idxNew], static_cast<int>(2))));
				tempVar[3][1] = wSelf * (mVarM2[idxNew] + varianceInterpolateParam * (pow(mMeanM2Tmp[idxNow] - mMeanM2[idxNew], static_cast<int>(2))));
			}

			if (sumW > 0) {
				mVarM1Tmp[idxNow] = (tempVar[0][0] + tempVar[1][0] + tempVar[2][0] + tempVar[3][0]) / sumW;
				mVarM2Tmp[idxNow] = (tempVar[0][1] + tempVar[1][1] + tempVar[2][1] + tempVar[3][1]) / sumW;

				/*mVarM1Tmp[idxNow] = (tempVar[3][0]) / sumW;
				mVarM2Tmp[idxNow] = (tempVar[3][1]) / sumW;*/
			}

			//Limitations and exceptions
			mVarM1Tmp[i + j * modelWidth] = MAX(mVarM1Tmp[i + j * modelWidth], minBGVar);
			mVarM2Tmp[i + j * modelWidth] = MAX(mVarM2Tmp[i + j * modelWidth], minBGVar);

			if (idxNewI < 1 || idxNewI >= modelWidth - 1 || idxNewJ < 1 || idxNewJ >= modelHeight - 1) {
				mVarM1Tmp[i + j * modelWidth] = initBGVar;
				mVarM2Tmp[i + j * modelWidth] = initBGVar;
				mAgeM1Tmp[i + j * modelWidth] = 0;
				mAgeM2Tmp[i + j * modelWidth] = 0;
			}
			else {
				mAgeM1Tmp[i + j * modelWidth] = MIN(mAgeM1Tmp[i + j * modelWidth] * exp(-varDecRatio * MAX(0.0, mVarM1Tmp[i + j * modelWidth] - varMinNoiseT)), maxBGAge);
				mAgeM2Tmp[i + j * modelWidth] = MIN(mAgeM2Tmp[i + j * modelWidth] * exp(-varDecRatio * MAX(0.0, mVarM2Tmp[i + j * modelWidth] - varMinNoiseT)), maxBGAge);
			}
		}
	}
}

void ProbModel::update(Mat& outputImg) {
	outputImg.setTo(Scalar(0, 0, 0));
	//unsigned char* pOut = (unsigned char*)outputImg.data;
	//unsigned char* pCur = (unsigned char*)frameGray.data;

	////--------------------------------------------
	//int widthTmp = frameGray.cols;
	//int nchannelsTmp = frameGray.channels();

	//int obsWidthStep = ((widthTmp * sizeof(unsigned char) * nchannelsTmp) % 4 != 0) ? ((((widthTmp * sizeof(unsigned char) * nchannelsTmp) / 4) * 4) + 4) : (widthTmp * sizeof(unsigned char) * nchannelsTmp);
	////---------------------------------------------
	//int obsWidthStep = frameGray.step; //check for padding

	//Find matching model
	memset(mModelID, 0, sizeof(int) * modelHeight * modelWidth);

	for (int bIdxJ{ 0 }; bIdxJ < modelHeight; bIdxJ++) {
		for (int bIdxI{ 0 }; bIdxI < modelWidth; bIdxI++) {
			//Base (i, j) for this block
			int idxBaseI = static_cast<float>(bIdxI) * blockSize;
			int idxBaseJ = static_cast<float>(bIdxJ) * blockSize;

			float curMean{ 0 };
			float elemCnt{ 0 };
			for (int jj{ 0 }; jj < blockSize; jj++) {
				for (int ii{ 0 }; ii < blockSize; ii++) {
					int idxI{ idxBaseI + ii };
					int idxJ{ idxBaseJ + jj };

					if (idxI < 0 || idxI >= obsWidth || idxJ < 0 || idxJ >= obsHeight)
						continue;

					//curMean += pCur[idxI + idxJ * obsWidthStep];
					curMean += frameGray.at<unsigned char>(idxJ, idxI);
					elemCnt += 1.0;
				}
			}
			curMean /= elemCnt;

			//Make oldest Idx to 0 (swap)
			int oldIdx{ 0 };
			int oldAge{ 0 };
			
			float fAge{ mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] };
			if (fAge >= oldAge) {
				oldIdx = 0;
				oldAge = fAge;
			}

			fAge = mAgeM2Tmp[bIdxI + bIdxJ * modelWidth];
			if (fAge >= oldAge) {
				oldIdx = 1;
				oldAge = fAge;
			}

			if (oldIdx != 0) {
				mMeanM1Tmp[bIdxI + bIdxJ * modelWidth] = (oldIdx == 0) ? mMeanM1Tmp[bIdxI + bIdxJ * modelWidth] : mMeanM2Tmp[bIdxI + bIdxJ * modelWidth];
				if (oldIdx == 0)
					mMeanM1Tmp[bIdxI + bIdxJ * modelWidth] = curMean;
				else
					mMeanM2Tmp[bIdxI + bIdxJ * modelWidth] = curMean;

				mVarM1Tmp[bIdxI + bIdxJ * modelWidth] = (oldIdx == 0) ? mVarM1Tmp[bIdxI + bIdxJ * modelWidth] : mVarM2Tmp[bIdxI + bIdxJ * modelWidth];
				if (oldIdx == 0)
					mVarM1Tmp[bIdxI + bIdxJ * modelWidth] = initBGVar;
				else
					mVarM2Tmp[bIdxI + bIdxJ * modelWidth] = initBGVar;

				mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] = (oldIdx == 0) ? mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] : mAgeM2Tmp[bIdxI + bIdxJ * modelWidth];
				if (oldIdx == 0)
					mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] = 0;
				else
					mAgeM2Tmp[bIdxI + bIdxJ * modelWidth] = 0;
			}

			//Select Model
			//Check match against 0
			if (pow(curMean - mMeanM1Tmp[bIdxI + bIdxJ * modelWidth], static_cast<int>(2) < varThreshModelMatch * mVarM1Tmp[bIdxI + bIdxJ * modelWidth]))
				mModelID[bIdxI + bIdxJ * modelWidth] = 0;
			//Check match against 1
			else if (pow(curMean - mMeanM2Tmp[bIdxI + bIdxJ * modelWidth], static_cast<int>(2) < varThreshModelMatch * mVarM2Tmp[bIdxI + bIdxJ * modelWidth]))
				mModelID[bIdxI + bIdxJ * modelWidth] = 1;
			//If no match, set 2 age to zero and match = 1
			else {
				mModelID[bIdxI + bIdxJ * modelWidth] = 1;
				mAgeM2Tmp[bIdxI + bIdxJ * modelWidth] = 0;
			}
		}
	}

	//Update with current observation
	float obsMean[2];
	float obsVar[2];

	for (int bIdxJ{ 0 }; bIdxJ < modelHeight; bIdxJ++) {
		for (int bIdxI{ 0 }; bIdxI < modelWidth; bIdxI++) {
			//Base (i, j) for this block
			int idxBaseI = static_cast<float>(bIdxI) * blockSize;
			int idxBaseJ = static_cast<float>(bIdxJ) * blockSize;

			int nMatchIdx{ mModelID[bIdxI + bIdxJ * modelWidth] };

			//Obtain observation mean
			memset(obsMean, 0, sizeof(float) * numModels);
			int nElemCnt[2];
			memset(nElemCnt, 0, sizeof(int) * numModels);
			for (int jj{ 0 }; jj < blockSize; jj++) {
				for (int ii{ 0 }; ii < blockSize; ii++) {
					int idxI{ idxBaseI + ii };
					int idxJ{ idxBaseJ + jj };
					if (idxI < 0 || idxI >= obsWidth || idxJ < 0 || idxJ >+obsHeight)
						continue;

					//obsMean[nMatchIdx] += pCur[idxI + idxJ * obsWidthStep];
					obsMean[ nMatchIdx ] += frameGray.at<unsigned char>(idxJ, idxI);
					++nElemCnt[nMatchIdx];
				}
			}
			
			//Model 1
			if (nElemCnt[0] <= 0)
				mMeanM1[bIdxI + bIdxJ * modelWidth] = mMeanM1Tmp[bIdxI + bIdxJ * modelWidth];
			else {
				//Learning rate for this block
				float age{ mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] };
				float alpha = age / (age + 1.0);

				obsMean[0] /= static_cast<float>(nElemCnt[0]);
				//Update with this mean
				if (age < 1)
					mMeanM1[bIdxI + bIdxJ * modelWidth] = obsMean[0];
				else
					mMeanM1[bIdxI + bIdxJ * modelWidth] = alpha * mMeanM1Tmp[bIdxI + bIdxJ * modelWidth] + (1.0 - alpha) * obsMean[0];
			}

			//Model 2
			if (nElemCnt[1] <= 0)
				mMeanM2[bIdxI + bIdxJ * modelWidth] = mMeanM2Tmp[bIdxI + bIdxJ * modelWidth];
			else {
				//Learning rate for this block
				float age{ mAgeM2Tmp[bIdxI + bIdxJ * modelWidth] };
				float alpha = age / (age + 1.0);

				obsMean[0] /= static_cast<float>(nElemCnt[0]);
				//Update with this mean
				if (age < 1)
					mMeanM2[bIdxI + bIdxJ * modelWidth] = obsMean[0];
				else
					mMeanM2[bIdxI + bIdxJ * modelWidth] = alpha * mMeanM2Tmp[bIdxI + bIdxJ * modelWidth] + (1.0 - alpha) * obsMean[1];
			}
		}
	}

	for (int bIdxJ{ 0 }; bIdxJ < modelHeight; bIdxJ++) {
		for (int bIdxI{ 0 }; bIdxI < modelWidth; bIdxI++) {
			//Base (i,j) for this block
			int idxBaseI = static_cast<float>(bIdxI) * blockSize;
			int idxBaseJ = static_cast<float>(bIdxJ) * blockSize;
			int nMatchIdx{ mModelID[bIdxI + bIdxJ * modelWidth] };

			//Obtain observation variance
			memset(obsVar, 0, sizeof(float)* numModels);
			int nElemCnt[2];
			memset(nElemCnt, 0, sizeof(int)* numModels);

			for (int jj{ 0 }; jj < blockSize; jj++) {
				for (int ii{ 0 }; ii < blockSize; ii++) {
					int idxI{ idxBaseI + ii };
					int idxJ{ idxBaseJ + jj };
					nElemCnt[nMatchIdx]++;

					if (idxI < 0 || idxI >= obsWidth || idxJ < 0 || idxJ >= obsHeight)
						continue;

					float pixelDist{ 0.0 };
					//float fDiff{ (nMatchIdx == 0) ? pCur[idxI + idxJ * obsWidthStep] - mMeanM1[bIdxI + bIdxJ * modelWidth] : pCur[idxI + idxJ * obsWidthStep] - mMeanM2[bIdxI + bIdxJ * modelWidth] };
					float fDiff{ (nMatchIdx == 0) ? frameGray.at<unsigned char>(idxJ, idxI) - mMeanM1[bIdxI + bIdxJ * modelWidth] : frameGray.at<unsigned char>(idxJ, idxI) - mMeanM2[bIdxI + bIdxJ * modelWidth] };
					pixelDist += pow(fDiff, static_cast<int>(2));

					mDistImg.at<unsigned char>(idxJ, idxI) = pow(frameGray.at<unsigned char>(idxJ, idxI) - mMeanM1[bIdxI + bIdxJ * modelWidth], static_cast<int>(2));

					if (mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] > 1) {
						unsigned char valOut = mDistImg.at<unsigned char>(idxJ, idxI) > varThreshFGDetermine * mVarM1Tmp[bIdxI + bIdxJ * modelWidth] ? 255 : 0;
						//pOut[idxI + idxJ * obsWidthStep] = valOut;
						outputImg.at<unsigned char>(idxI, idxJ);
					}

					obsVar[nMatchIdx] = MAX(obsVar[nMatchIdx], pixelDist);
				}
			}

			if (nElemCnt[0] > 0) {
				float age{ mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] };
				float alpha = age / (age + 1.0);

				//update with this variance
				if (age == 0)
					mVarM1[bIdxI + bIdxJ * modelWidth] = MAX(obsVar[0], initBGVar);
				else {
					float alphaVar{ alpha };
					mVarM1[bIdxI + bIdxJ * modelWidth] = alphaVar * mVarM1Tmp[bIdxI + bIdxJ * modelWidth] + (1.0 - alphaVar) * obsVar[0];
					mVarM1[bIdxI + bIdxJ * modelWidth] = MAX(mVarM1[bIdxI + bIdxJ * modelWidth], minBGVar);
				}

				//Update age
				mAgeM1[bIdxI + bIdxJ * modelWidth] = mAgeM1Tmp[bIdxI + bIdxJ * modelWidth] + 1.0;
				mAgeM1[bIdxI + bIdxJ * modelWidth] = MIN(mAgeM1[bIdxI + bIdxJ * modelWidth], maxBGAge);
			}
			else {
				mVarM1[bIdxI + bIdxJ * modelWidth] = mVarM1Tmp[bIdxI + bIdxJ * modelWidth];
				mAgeM1[bIdxI + bIdxJ * modelWidth] = mAgeM1Tmp[bIdxI + bIdxJ * modelWidth];
			}

			if (nElemCnt[1] > 0) {
				float age{ mAgeM2Tmp[bIdxI + bIdxJ * modelWidth] };
				float alpha = age / (age + 1.0);

				//update with this variance
				if (age == 0)
					mVarM2[bIdxI + bIdxJ * modelWidth] = MAX(obsVar[0], initBGVar);
				else {
					float alphaVar{ alpha };
					mVarM2[bIdxI + bIdxJ * modelWidth] = alphaVar * mVarM2Tmp[bIdxI + bIdxJ * modelWidth] + (1.0 - alphaVar) * obsVar[0];
					mVarM2[bIdxI + bIdxJ * modelWidth] = MAX(mVarM2[bIdxI + bIdxJ * modelWidth], minBGVar);
				}

				//Update age
				mAgeM2[bIdxI + bIdxJ * modelWidth] = mAgeM2Tmp[bIdxI + bIdxJ * modelWidth] + 1.0;
				mAgeM2[bIdxI + bIdxJ * modelWidth] = MIN(mAgeM2[bIdxI + bIdxJ * modelWidth], maxBGAge);
			}
			else {
				mVarM2[bIdxI + bIdxJ * modelWidth] = mVarM2Tmp[bIdxI + bIdxJ * modelWidth];
				mAgeM2[bIdxI + bIdxJ * modelWidth] = mAgeM2Tmp[bIdxI + bIdxJ * modelWidth];
			}

		}
	}
}