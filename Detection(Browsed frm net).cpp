
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace cv;


int main(){

	VideoCapture cap(0); // open the video camera no. 0
	Mat frame, grayframe, framehsv, roi;
	vector <Mat>channels;
	int c = -1, d = -1;
	int j = 0;
	bool bSuccess;
	vector<Rect> faces;
	vector<Rect> eyes;
	Point2f center;
	// Load Face cascade (.xml file)
	CascadeClassifier face_cascade;
	CascadeClassifier eye_cascade;

	face_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml");
	eye_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");

	if (eye_cascade.empty()) {
		printf("EC Not loaded\n");
		return -10;
	}

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;

//	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	namedWindow("Detected Face", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"


	while (1)
	{

		cap >> frame; // read a new frame from video
		cvtColor(frame, grayframe, COLOR_BGR2GRAY);
		// Detect faces
		faces.clear();
		face_cascade.detectMultiScale(grayframe, faces, 1.1, 3, CV_HAAR_SCALE_IMAGE, Size(30, 30));
		// Draw circles on the detected faces
		imshow("MyVideo", frame);

		for (int i = 0; i < faces.size(); i++){

			eyes.clear();
			roi = grayframe(faces[i]);
			eye_cascade.detectMultiScale(roi,eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

			for (j = 0; j < eyes.size(); j++) {
				center = Point2f{ (float)(faces[i].x + eyes[j].x + eyes[j].width*0.5), 
								(float)(faces[i].y + eyes[j].y + eyes[j].height*0.5) };
				circle(frame, center, (eyes[j].width + eyes[j].height)*0.25, Scalar(255, 0, 0), 4, 8, 0);
//				rectangle(frame, eyes[j], Scalar(0, 0, 255), 2, 8, 0);
			}
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
		}

		imshow("Detected Face", frame);

		c = waitKey(0);
		printf("displaying\n");
		if (c == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			//	cout << "esc key is pressed by user" << endl;
			printf("Esc is pressed.");
			break;
		}


	}

	face_cascade.~CascadeClassifier();
	faces.clear();
	return 0;
}


