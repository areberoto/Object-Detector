#include "MC.h"

MC::MC() : lucasKanade{} {

}

void MC::init(const Mat& src) {
	frame = src;
	Mat tmp;
	cvtColor(frame, tmp, COLOR_BGR2GRAY);
	medianBlur(tmp, frameGray, 5);

	lucasKanade.init(frameGray);
	BGmodel.init(frameGray);
	frameGrayPrev = frameGray.clone();
}

void MC::run() {
	Mat tmp;
	cvtColor(frame, tmp, COLOR_BGR2GRAY);
	medianBlur(tmp, frameGray, 5);

	lucasKanade.runTrack(frameGray);
	H = lucasKanade.getHomography().clone();
	std::cout << H << std::endl;
	BGmodel.motionCompensate(H);
	BGmodel.update(detectImg);
	frameGrayPrev = frameGray.clone();
	waitKey(10);
}