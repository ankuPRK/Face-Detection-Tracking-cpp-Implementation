/*
Utilised these two cpp files to write code for basic detection+tracking :
1)	http://opencvexamples.blogspot.com/2013/10/face-detection-using-haar-cascade.html (Viola Jones algorithm for Face Detection)
2)	opencv\sources\samples\cpp\lkdemo.cpp (Lucas Kannade Algorithm for tracking)


*/


/////////libraries//////////
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>

//////using namespace/////////
using namespace cv;
using namespace std;

////functions/////
char *get_string_id(int minutecount, int faceid, int secondcount, int tf, char path[]);
Point2f get_centroid(vector<Point2f> *ppoints);
Point2f get_variance(vector<Point2f> *ppoints, Point2f centroid);

int main(int argc, char** argv)
{

	//	VideoCapture cap("G:\\vids\\friends_clip.mp4");	//Capture video from Camera. To capture a video file replace '0' with the full path directory. 
	VideoCapture cap(0);
	// Replace all the '\' in path with '\\'
	if (!cap.isOpened())	// if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	Mat image, frameRaw, frame, croppedimg;			//type of Mat will be CV_8U 3 channels
	Mat grayRaw, grayF, prevGray, mask;   //type of Mat will be CV_8U 1 channel

	////// Load Face cascade (.xml file) for face detection /////
	CascadeClassifier face_cascade;
	face_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml");
	vector<Rect> faces;		//vector for storing all the faces in the form of vector of rectangles on the image. Each rectangle 
	//precisely bounds a single face.

	if (face_cascade.empty())	// if not success, exit program
	{
		cout << "Cannot load the xml file." << endl;
		return -2;
	}

	//Parameters of Detection of feature points on the first frame; see the definitions of functions GoodFeaturesToTrack
	//and CornerSubPix for more details//	
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	vector < vector<Point2f> > points;			//Contains a vector of "vector of points for each face", which are iteratively 
	//updated after each frame
	vector < vector<Point2f> > pointsPrev;		//vector of 'vector of points for each face' for the previous frame
	vector<Point2f>  tempPoints;
	const int MAX_COUNT = 500;					//max no. of points per rectangle

	//Parameters utilised in calculating the optical flow using the KLT algorithm//
	vector < vector<uchar> > status;		//status is 1 if the updated point is valid, else its status is 0.
	vector < vector<float> > err;
	vector < vector<Point2f> > tmp;
	vector<uchar>  tempStatus;
	vector<float>  tempErr;
	vector<Point2f>  tempTmp;
	Point2f topleft;
	Point2f Botright;


	//////parameters to calculate the new(updated) rectangles(faces)///////
	vector<Point2f> centroid;		//vector of centroids of each rectangle(face)
	vector<Point2f> dim;			//vector of (width, height) of the rectangles of previous frame

	///////parameters for controlling loop and saving image/////
	int i, j, k;
	bool needToDetect = true;
	int minutecount = 0;
	int secondcount = 0;
	int min_to_sec = 50;
	int nFaces = 0;
	char *str;
	Rect Cropper;
	bool bsuccess = false;
	//////height and width of the frame/////
	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	cout << "Frame size : " << dWidth << " x " << dHeight << endl;
	cout << "Press Esc to exit.";

	namedWindow("Detection+Tracking", WINDOW_AUTOSIZE);

	cap >> frameRaw;			// read first frame
	cvtColor(frameRaw, grayRaw, COLOR_BGR2GRAY);
	grayRaw.copyTo(mask);		//use it to give size and type to the mask

	system("md data");		//create directory names 'data'

	//////The while(1) loop//////
	/*
	The loop does the following:
	->	It detects the rectangles for each face and in each rectangle
	it detects good features using GoodFeaturesToTrack function.
	->	Then it just tracks those points by calculating their Optical Flow
	by using KLT algorithm via calcOpticalFlowPyrLK function.
	-	To draw rectangles on tracked points, we use two assumptions:
	a. Centroid of the updated points of each face will be the centroid of the updated rectangle.
	b. Ratio of the dimentions of the new rectangle to the old rectangle will be the ratio of the
	variance of the points of new rectangle to the variance of the points of the old rectangle
	->	After 50 frames, it refreshes and starts from again.
	->	The loop terminates if the user presses the Esc key.
	*/

	while (1)
	{
		cap >> frameRaw;			//read frame
		if (frameRaw.empty())
			break;
		GaussianBlur(frameRaw, frame, { 7, 7 }, 1.0, 1.0, BORDER_DEFAULT);
		frame.copyTo(image);	//copy it to image
		cvtColor(image, grayRaw, COLOR_BGR2GRAY);	//convert into grayscale
		equalizeHist(grayRaw, grayF);

		//This part of the loop detects the faces using Viola Jones Algorithm
		if (needToDetect) {
			// Detect faces
			faces.clear();
			face_cascade.detectMultiScale(grayF, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
			nFaces = faces.size();	//no of faces

			//preparing the vectors
			points.clear();
			centroid.resize(nFaces);
			dim.resize(nFaces);
			/////detectING the good features in all the rectangles and storing those features in points vector////
			for (i = 0; i < nFaces; i++) {

				//making the mask for ith rectangle
				mask = Scalar(0, 0, 0);
				mask(faces[i]) = Scalar(255);

				//store good features of ith rectangle in tempPoints vector
				tempPoints.clear();
				goodFeaturesToTrack(grayF, tempPoints, MAX_COUNT, 0.01, 10, mask, 3, 0, 0.04);
				points.push_back(tempPoints);	//push tempPoints to points
				cornerSubPix(grayF, points[i], subPixWinSize, Size(-1, -1), termcrit);


				if (points[i].empty()) {
//					printf("No point in this face during detection.\n");
					faces.erase(faces.begin() + i);
					points.erase(points.begin() + i);
					nFaces = faces.size();
					i--;
					centroid.resize(nFaces);
					dim.resize(nFaces);
				}
				else {
					//calculate the centroid
					centroid[i].x = faces[i].x + faces[i].width / 2.0;
					centroid[i].y = faces[i].y + faces[i].height / 2.0;

					//get the dimensions and centroids of all the rectangles
					dim[i].x = faces[i].width;
					dim[i].y = faces[i].height;

					//draw the points on the image
					for (j = 0; j < points[i].size(); j++) {
						cv::circle(image, points[i][j], 1, Scalar(255, 255, 255), -1, 8);
					}
				}
			}
			needToDetect = false;
//			printf("Detection of frame is completed successfully.\n");
		}
		else
		{
			////do the tracking////
			status.clear();
			err.clear();
			for (i = 0; i < nFaces; i++) {
				tempStatus.clear();
				tempErr.clear();
				//KLT tracking function
				calcOpticalFlowPyrLK(prevGray, grayF, pointsPrev[i], points[i], tempStatus, tempErr, winSize,
					3, termcrit, 0, 0.001);
				status.push_back(tempStatus);
				err.push_back(tempErr);
			}

			for (i = 0; i < nFaces; i++) {

				//update the points
				for (j = 0, k = 0; j < points[i].size(); j++)
				{
					if (!status[i][j])
						continue;
					points[i][k++] = points[i][j];
					cv::circle(image, points[i][j], 1, Scalar(255, 255, 255), -1, 8);
				}
				points[i].resize(k);
//				printf("Tracking points of this frame updated.\n");
				if (points[i].empty()) {
					faces.erase(faces.begin() + i);
					points.erase(points.begin() + i);
					status.erase(status.begin() + i);
					err.erase(err.begin() + i);
					dim.erase(dim.begin() + i);
					nFaces = faces.size();
					i--;
					centroid.resize(nFaces);
//					printf("One face erased during tracking.\n");
				}
				else {
					//calculate centroid
					centroid[i] = get_centroid(&(points[i]));

					//draw the centriod in red color
					cv::circle(image, centroid[i], 2, Scalar(0, 0, 255), -1, 8);
//					printf("Variance and Centroid found successfully during Tracking. \n");
				}
			}
		}

		//
		//draw rectangles around the faces
		for (i = 0; i < nFaces; i++) {

			//binding the rectangle into the image
			topleft.x = (centroid[i].x - dim[i].x / 2 > 0 ? centroid[i].x - dim[i].x / 2 : 0);
			topleft.y = (centroid[i].y - dim[i].y / 2 > 0 ? centroid[i].y - dim[i].y / 2 : 0);
			Botright.x = (centroid[i].x + dim[i].x / 2 < dWidth ? centroid[i].x + dim[i].x / 2 : dWidth);
			Botright.y = (centroid[i].y + dim[i].y / 2 < dHeight ? centroid[i].y + dim[i].y / 2 : dHeight);

//			cout << "topleft: " << topleft << "botright: " << Botright << "\n";
			Cropper = Rect{ (int)topleft.x, (int)topleft.y, (int)Botright.x - (int)topleft.x, (int)Botright.y - (int)topleft.y };

			rectangle(image, Cropper, Scalar(255, 0, 0), 2, 8, 0);
			str = get_string_id(minutecount, i, secondcount, 0, "data\\");
//			cout << "cropping rectangle: " << Cropper << "\n";
			frame(Cropper).copyTo(croppedimg);
			imwrite(str, croppedimg);
//			cout << "id: " << str << "\n";
		}

		//show the image in Window
		imshow("Detection+Tracking", image);

		//Wait for 10ms if a key is pressed. Note that even if you don't want any key-press input from the user still you 
		//have to keep the waitKey() function, because it is required by highGUI to handle the events of the window. 
		char c = (char)waitKey(50);
		if (c == 27) {
			break;		//exit if Esc is pressed
		}

		pointsPrev.clear();
		pointsPrev = points;		//update pointsPrev
		grayF.copyTo(prevGray);		//update the previous gray frame
		secondcount++;				//increment the frame counter

		if (secondcount == min_to_sec) {
			//start afresh
			pointsPrev.clear();
			needToDetect = true;
			secondcount = 0;
			minutecount++;
		}
	}

	return 0;
}


char *get_string_id(int minutecount, int faceid, int secondcount, int tf, char path[]) {

	int minutesize = 6;
	int secondsize = 3;
	int faceidsize = 2;
	int tfsize = 1;
	int idsize = minutesize + secondsize + faceidsize + tfsize + 4 + 1;		//4 is for extension .jpg and 1 is for '\0'
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
	for (i = 0; i < secondsize; i++) {
		out[minutesize + secondsize + faceidsize - i - 1 + length] = 48 + (secondcount % 10);
		secondcount = secondcount / 10;
	}
	for (i = 0; i < tfsize; i++) {
		out[tfsize + minutesize + secondsize + faceidsize - i - 1 + length] = 48 + (tf % 10);
		tf = tf / 10;
	}
	out[length + idsize - 1] = '\0';
	out[length + idsize - 2] = 'g';
	out[length + idsize - 3] = 'p';
	out[length + idsize - 4] = 'j';
	out[length + idsize - 5] = '.';

	return out;
}

//function to calculate the variance of a Point2f vector, having a Point2f centroid
Point2f get_variance(vector<Point2f> *ppoints, Point2f centroid) {
	int j = 0;
	Point2f variance;
	variance.x = 0;
	variance.y = 0;
	if ((*ppoints).size() == 0) {
		printf("Variance: No points found.\n");
		return (Point2f)NULL;
	}
	for (j = 0; j < (*ppoints).size(); j++) {
		variance.x = variance.x + ((*ppoints)[j].x - centroid.x)*((*ppoints)[j].x - centroid.x);
		variance.y = variance.y + ((*ppoints)[j].y - centroid.y)*((*ppoints)[j].y - centroid.y);
	}
	variance.x = variance.x / (*ppoints).size();
	variance.y = variance.y / (*ppoints).size();
	return variance;
}

//function to calculate the centroid of a Point2f vector
Point2f get_centroid(vector<Point2f> *ppoints) {
	int p = 0;
	Point2f centroid;
	centroid.x = 0;
	centroid.y = 0;

	if ((*ppoints).size() == 0) {
		printf("Centroid: No points found.\n");
		return (Point2f)NULL;
	}

	for (int p = 0; p < (*ppoints).size(); p++) {
		centroid.x = centroid.x + (*ppoints)[p].x;
		centroid.y = centroid.y + (*ppoints)[p].y;
	}
	centroid.x = centroid.x / (*ppoints).size(); //Mean cordinates of points[i]
	centroid.y = centroid.y / (*ppoints).size(); // Mean cordinates of points[i]

	return centroid;
}
