#include "KLT.h"

KLT::KLT() : winSize{ 10 }, status{}, count{ 0 }, flags{ 0 }, p0{}, p1{} {
	H = (Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); //Identity

}

void KLT::init(const Mat& imgGray) {
	H = (Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); //Identity
	maskImg = Mat(imgGray.cols, imgGray.rows, CV_8UC1, Scalar(255));
}

void KLT::initFeatures(Mat imgGray) {
	count = imgGray.cols / 32 * imgGray.rows / 24;

	for (size_t i{ 0 }; i < 10; i++) {
		for (size_t j{ 0 }; j < 10; j++)
			p1.push_back(Point2f(i * 32 + 16, j * 24 + 12));
	}

	swapData(imgGray);
}

void KLT::runTrack(Mat frameGray) {
	vector<int> nMatch{};

	Mat prevGray;
	if (prevGray.empty())
		prevGray = frameGrayPrev;
	else
		flags = 0;

	if (count > 0) {
		
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::MAX_ITER)+(TermCriteria::EPS), 20, 0.03);
		calcOpticalFlowPyrLK(prevGray, frameGray, p0, p1, status, err, Size(winSize, winSize), 3, criteria, flags);
		
		for (size_t i{ 0 }; i < count; i++) {
			if (!status[i])
				continue;
			nMatch.push_back(i);
		}
		count = nMatch.size();
	}

	if (count >= 10)
		makeHomography(nMatch);
	else
		H = (Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); //Identity

	//std::cout << H << std::endl;
	initFeatures(frameGray);
}

void KLT::swapData(Mat imgGray) {
	frameGrayPrev = imgGray.clone();
	swap(p0, p1);
}

void KLT::makeHomography(vector<int> nMatch) {
	vector<Point2f> p1T{}, p0T{};

	for (size_t i{ 0 }; i < count; i++) {
		p1T.push_back(p1.at(nMatch[i]));
		p0T.push_back(p0.at(nMatch[i]));
	}

	Mat pt1(1, count, CV_32FC2, p1T.data());
	Mat pt0(1, count, CV_32FC2, p0T.data());
	//std::cout << pt1 << std::endl;
	H = findHomography(pt1, pt0, RANSAC, 1);
	H.convertTo(H, CV_32FC1);
}

Mat KLT::getHomography() {
	Mat tmp = H.clone();
	//std::cout << tmp << std::endl;
	return tmp.clone();
}