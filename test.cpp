

#include <iostream>  
#include <unordered_map>
#include <string.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc.hpp>


#define PI 3.1415926
#define WINDOW_SIZE 128
using namespace cv;
using namespace std;

Point rescale(Point point, Point origin, double scale);
Point rotate(Point point, Point origin, double angle);
Point rescaleAndRotate(Point point, Point origin, double scale, double angle);

class Components {
	/*
		x, y:	center of the components
		type:	component type (1, 2, 3...)
		scale:	a rescale coefficient
		orientation:	an angle it is rotated
	*/
	protected:
		int x;
		int y;
		int type;
		double scale;
		double orientation;

	public:
		Components(int x, int y, int type, double scale, double orientation) {
			this->x = x;
			this->y = y;
			this->type = type;
			this->scale = scale;
			this->orientation = orientation;
		}

		//	get/set methods
		void setX(int x){
			this->x = x;
		}
		void setY(int y){
			this->y = y;
		}
		void setType(int type) {
			this->type = type;
		}
		void setScale(double scale) {
			this->scale = scale;
		}
		void setOrientation(double orientation) {
			this->orientation = orientation;
		}
		int getX() {
			return this->x;
		}
		int getY() {
			return this->y;
		}
		int getType() {
			return this->type;
		}
		double getScale() {
			return this->scale;
		}
		double getOrientation() {
			return this->orientation;
		}

		virtual Mat drawPic() {
			return Mat();
		};
};

class Face : public Components {
	int w = WINDOW_SIZE;
	public:
		Face(int x, int y, int type, double scale, double orientation) : Components(x, y, type, scale, orientation) {
			cout << "Face Initialized!";
		};
		Mat drawPic(Mat panel) {

			if (type == 1) {
				// Ellipsoid
				ellipse(panel, Point(x, y), Size(int(w / 4 * scale), int(w / 3.5 * scale)), orientation, 0, 360, Scalar(0));
			}
			if (type == 2) {
				// ×¶×ÓÁ³
				vector<Point> pointLists;
				pointLists.push_back(Point(x - 32, y - 32));
				pointLists.push_back(Point(x - 24, y + 16));
				pointLists.push_back(Point(x, y + 32));
				pointLists.push_back(Point(x + 24, y + 16));
				pointLists.push_back(Point(x + 32, y - 32));
				Point origin = Point(x, y);

				vector<Point> newPointLists;
				for (int i = 0; i < pointLists.size(); i++) {
					newPointLists.push_back(rescaleAndRotate(pointLists[i], Point(x, y), scale, orientation));
				}
				polylines(panel, newPointLists, true, Scalar(0), 1, 8, 0);
			}
			return panel;
		}
};

/*
	Inputs:	
		point: point that needed to be rescaled
		orginal: the original point
		scale:	scale coefficient;
	Return:
		a point rescaled.
*/
/*
Point rescale(Point point, Point origin, double scale) {
	Point diff = Point(point.x - origin.x, point.y - origin.y);
	double diff_x = diff.x * scale;
	double diff_y = diff.y * scale;
	return Point(round(origin.x + diff_x), round(origin.y + diff_y));
}
*/
/*
	Inputs:
		point: point that needed to be rotated
		orginal: the original point
		angle:	the angle rotated (clock-wise)
	Return:
		a point rotated
*/

/*
Point rotate(Point point, Point origin, double angle) {
	int x = point.x - origin.x;
	int y = point.y - origin.y;
	double distance = sqrt(x * x + y * y);
	double theta = atan2(y, x) * 180 / PI;
	double theta_new = theta + angle;

	double new_x = distance * cos(theta_new * PI / 180);
	double new_y = distance * sin(theta_new * PI / 180);
	return Point(round(origin.x + new_x), round(origin.y + new_y));
}
*/
/*
Point rescaleAndRotate(Point point, Point origin, double scale, double angle) {
	Point A = rescale(point, origin, scale);
	Point B = rotate(point, origin, angle);
	return B;
}
*/
/*
int main()
{
	// given a vector (Face, Left-Eye, Right-Eye, Left-Ear, Right-Ear, Nose, Mouth)
	
	Face face1 = Face(64, 64, 2, 1, 10);
	Mat panel = Mat(WINDOW_SIZE, WINDOW_SIZE, CV_8UC1, Scalar(255));
	panel = face1.drawPic(panel);
	//Components Face = Components();

	String windowName = "MyWindow";
	imshow(windowName, panel);
	waitKey(0);
	
	//cout << rescale(Point(4, -3), Point(0, 0), 3);

}
*/
