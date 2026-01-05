#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    int numImgs = 5;

    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints;

    vector<Mat> images;
    images.reserve(numImgs);

    String base1 = "";
    String base2 = "";
    
    for(int i = 0; i < numImgs; i++) {

        String p = 

    }
    findChessboardCorners
    return 0;
}