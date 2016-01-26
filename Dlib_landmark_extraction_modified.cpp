// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

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
using namespace dlib;

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        image_window win;
		char c = 'z';
        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
		cout << "start reading shape_predictor_68_face_landmarks.dat \t";
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		cout << "Reading is done. \n";

		////// Load Face cascade (.xml file) for face detection /////
		CascadeClassifier face_cascade;
		face_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml");
		std::vector<Rect> faces;		//vector for storing all the faces in the form of vector of rectangles on the image. Each rectangle 
		//precisely bounds a single face.
		std::vector<point> *val;
		if (face_cascade.empty())	// if not success, exit program
		{
			cout << "Cannot load the xml file." << endl;
			return -2;
		}

		namedWindow("video", WINDOW_AUTOSIZE);

		cv::Mat temp;
		std::vector <dlib::rectangle> facesR;
		std::vector <dlib::full_object_detection> shapes;
		int i = 0,j=0;
		Point sss = { 0, 0 };
		// Grab and process frames until the main window is closed by the user.
		while (1)
		{
			// Grab a frame
			cap >> temp;
			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.
			cv_image<bgr_pixel> cimg(temp);
			// Detect faces 
//			faces.clear();
			facesR.clear();
			shapes.clear();

			/*			face_cascade.detectMultiScale(temp, faces, 1.1, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
				// Find the pose of each face.
			facesR.resize(faces.size());
			for (i = 0; i < faces.size(); i++){
				facesR[i].set_left(faces[i].x);
				facesR[i].set_top(faces[i].y);
				facesR[i].set_right(faces[i].x + faces[i].width);
				facesR[i].set_bottom(faces[i].y + faces[i].height);
			}
*/			
			// Detect faces 
			facesR = detector(cimg);

			// Find the pose of each face.
			for (i = 0; i < facesR.size(); ++i){
				shapes.push_back(pose_model(cimg, facesR[i]));
			}
			for (i = 0; i < facesR.size(); i++){
				val = shapes[i].get_parts();
				for (j = 0; j < (*val).size(); j++){
					sss.x = (int)(*val)[j].x();
					sss.y = (int)(*val)[j].y();
					
					circle(temp, sss, 1, Scalar{ 255, 255, 255 }, 1, 8, 0);
				}
				//				cout << "size is : " << (*val).size();
			}
//			printf("\ndnn\n");
            // Display it all on the screen

			imshow("video", temp);
			c = waitKey(1);
			if (c == 27){
				break;
			}
/*			win.clear_overlay();
            win.set_image(cimg);
           win.add_overlay(render_face_detections(shapes));
*///			cout << "chal rio chhe \n";

		}
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }

//	getchar();
	return 0;
}

