/////////libraries//////////
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

//macros//
#define M_PI 3.141593

//////using namespace/////////
using namespace cv;
using namespace std;

////functions/////
char *get_string_id(int minutecount, int faceid, char path[]);
void OLBP(cv::Mat *psrc, cv::Mat *pdst);

int main(){
	Mat imgRaw,imgResized, imgGray;
	char path[] = "data\\";
	char path2[] = "data2\\";
	char *inAdd;
	char *outAdd;
	Size sz = { 102, 102 };
	int i, j;

	system("md data2");		//create directory names 'data2'


	for (i = 0; i < 207; i++){
		for (j = 0; j <= 1; j++){
			inAdd = get_string_id(i, j, path);
			imgRaw = imread(inAdd);
			if (!imgRaw.empty()){
				cout << inAdd;
				resize(imgRaw, imgResized, sz);
				cvtColor(imgResized, imgGray, CV_BGR2GRAY);
				Mat imgTx = Mat::zeros(imgGray.rows - 2, imgGray.cols - 2, CV_8UC1);
				OLBP(&imgGray, &imgTx);
				outAdd = get_string_id(i, j, path2);
				imwrite(outAdd, imgTx);
				cout << "\tdone.";
			}
		}
	}

	return 0;
}

char *get_string_id(int minutecount, int faceid, char path[]) {

	int minutesize = 6;
	int faceidsize = 2;
	int idsize = minutesize + faceidsize + 4 + 1;		//4 is for extension .jpg and 1 is for '\0'
	int i = 0;
	int length = 0;
	bool isCreated = false;
	while (path[i] != '\0') {
		length++;
		i++;
	}

	char *out = (char *)malloc(sizeof(char) * (idsize + length));

	for (i = 0; i < length; i++) {
		out[i] = path[i];
	}
	for (i = 0; i < minutesize; i++) {
		out[minutesize - i - 1 + length] = 48 + (minutecount % 10);
		minutecount = minutecount / 10;
	}
	for (i = 0; i < faceidsize; i++) {
		out[minutesize + faceidsize - i - 1 + length] = 48 + (faceid % 10);
		faceid = faceid / 10;
	}

	out[length + idsize - 1] = '\0';
	out[length + idsize - 2] = 'g';
	out[length + idsize - 3] = 'p';
	out[length + idsize - 4] = 'j';
	out[length + idsize - 5] = '.';

	return out;
}

void OLBP(cv::Mat *psrc, cv::Mat *pdst) {
	Mat src = *psrc;
	Mat dst = *pdst;
	for (int i = 1; i<src.rows - 1; i++) {
		for (int j = 1; j<src.cols - 1; j++) {
			unsigned char center = src.at<char>(i, j);
			unsigned char code = 0;
			code |= (src.at<unsigned char>(i - 1, j - 1) > center) << 7;
			code |= (src.at<unsigned char>(i - 1, j) > center) << 6;
			code |= (src.at<unsigned char>(i - 1, j + 1) > center) << 5;
			code |= (src.at<unsigned char>(i, j + 1) > center) << 4;
			code |= (src.at<unsigned char>(i + 1, j + 1) > center) << 3;
			code |= (src.at<unsigned char>(i + 1, j) > center) << 2;
			code |= (src.at<unsigned char>(i + 1, j - 1) > center) << 1;
			code |= (src.at<unsigned char>(i, j - 1) > center) << 0;
			dst.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}
