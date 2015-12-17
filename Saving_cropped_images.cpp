#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;
char *get_string_id(int minutecount, int faceid, int secondcount, int tf, char path[]);

int main(){

	VideoCapture cap(0); // open the video camera no. 0
	Mat frame, grayframe, framehsv,croppedframe;
	int c = -1, d = -1;
	bool bSuccess;
	Rect cropper;
	int id;
	int i, j, k;
	int tf = 0;
	int minutecount = 0;
	int secondcount = 0;
	char *str;

	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	face_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml");

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;
	cropper = Rect{ 10,10,(int)(dWidth / 2),(int)(dHeight / 2) };

	//	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	namedWindow("CroppedVideo", CV_WINDOW_AUTOSIZE);

	while (1)
	{

		cap >> frame; // read a new frame from video
		cvtColor(frame, grayframe, COLOR_BGR2GRAY);
		rectangle(frame, cropper, Scalar{ 255,0,0 }, 1, 8, 0);
		frame(cropper).copyTo(croppedframe);
		imshow("MyVideo", frame);
		imshow("CroppedVideo", croppedframe);
		str = get_string_id(minutecount, 7,secondcount, 0,"data\\");
		printf("%s\n", str);
		imwrite(str, croppedframe);
		free(str);
		c = waitKey(10);
		if (c == 27){		//wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
			//	cout << "esc key is pressed by user" << endl;
			printf("Esc is pressed.");
			break;
		}
		secondcount++;
		if (secondcount == 50) {
			secondcount = 0;
			minutecount++;
		}

	}

	return 0;
}
