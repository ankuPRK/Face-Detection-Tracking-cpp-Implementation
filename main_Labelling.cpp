/////////libraries//////////
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <iostream>
#include <stdio.h>

//macros//
#define M_PI 3.141593

//////using namespace/////////
using namespace cv;
using namespace std;

////functions/////
char *get_string_id(int minutecount, int faceid, char path[]);

int main(){

	Mat imgRaw;
	char path[] = "data\\";
	char *inAdd;
	int i, j;
	FILE *fp = fopen("label.txt", "w");
	char c = 't';

	namedWindow("Windowaa", WINDOW_AUTOSIZE);

	for (i = 0; i <= 206; i++){
		for (j = 0; j <= 1; j++){
			inAdd = get_string_id(i, j, path);

			imgRaw = imread(inAdd);
			if (!imgRaw.empty()){
				cout << "0 or 1: ";
				imshow("Windowaa", imgRaw);
				c = 't';
				while (1){
					c = waitKey(10);
					if (c == '0'){
						fwrite("0", sizeof(char), 2, fp);
						cout << "0\n";
						break;
					}
					else if (c == '1'){
						fwrite("1", sizeof(char), 2, fp);
						cout << "0\n";
						break;
					}
				}
			}
		}
	}
	fclose(fp);
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
