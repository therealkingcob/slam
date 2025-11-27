#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

struct feature {
    int x;
    int y;
    float angle;
};

vector <vector<int>> offsets = {{0 ,4},
                                    {1, 3},
                                    {2, 2},
                                    {3, 1},
                                    {4, 0},
                                    {3, -1},
                                    {2, -2},
                                    {1, -3},
                                    {0, -4},
                                    {-1, -3},
                                    {-2, -2},
                                    {-3, -1},
                                    {-4 ,0},
                                    {-3, 1},
                                    {-2, 2},
                                    {-1, 3},
                                    {0 ,4},
                                    {1, 3},
                                    {2, 2},
                                    {3, 1},
                                    {4, 0},
                                    {3, -1},
                                    {2, -2},
                                    {1, -3},
                                    {0, -4},
                                    {-1, -3},
                                    {-2, -2},
                                    {-3, -1},
                                    {-4 ,0},
                                    {-3, 1},
                                    {-2, 2},
                                    {-1, 3}
                                           };

int getBrightness(Mat im, int x, int y) {
    uchar b = im.at<uchar>(y,x);
    int brightness = static_cast<int>(b);
    return brightness;
    
}

float getOrientation(Mat im, int x, int y) {
    int r = 15;

    // getting all of the pixels in radius 15 circle around the main one



    int m10 = 0;
    int m01 = 0;

    for(int i = -r; i <= r; i++) {
        for(int j = -r; j <= r; j++) {
            int xx = i + x;
            int yy = j + y;

            if(xx < 0 || yy < 0 || xx >= im.cols || yy >= im.rows) {
                continue;
            }

            uchar I = im.at<uchar>(yy, xx);

            m10 += i * I;
            m01 += j * I;
        }
    }

    float angle = atan2((float) m01, (float) m10); //randians
    return angle;
}

vector<feature> findFeatures(Mat image) {
    int width = image.cols;
    int height = image.rows;

    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    vector <feature> features;

    int n = 12;
    //cout << "before the for loop" << endl;

    for (int i = 4; i < width-4; i++) {
        for(int j = 4; j < height-4; j++) {
            int b = getBrightness(gray_image, i, j);
            //cout << b << endl;

            //int x_offset = 0; 
            //int y_offset = 3;

            int cons = 0;

            double tolerance = 2.4;
            

            //cout << "before the loop" << endl;

            //we need to go 16 pixels for the entire radius plus n since the series to feature pixel candidates could start on the last pixel of the 16
            for(int f = 0; f < 16 + n; f++) {

                //cout << "inside the second loop" << endl;
                //this means that the brightess is good - makes it a candidate for a feature
                if(getBrightness(gray_image, i + offsets[f][0], j - offsets[f][1]) <  ((double) (b / tolerance)) || getBrightness(gray_image, i + offsets[f][0], j - offsets[f][1]) > ((double) (b * tolerance))) {
                    cons++;

                    //if we have reached the required amount of pixels that are valid  we leave the loop

                    //cout << "inside the brightness if statement" << endl;
                    

                } else {
                    cons = 0;
                }

                if(cons >= n) {

                        float angle = getOrientation(gray_image, i, j);
                        features.push_back({i, j, angle});

                        //cout << "inside second if statement" << endl;
                        break;
                    }
                    
            }

            //cout << "after the second loop" << endl;
        }
    }

    return features;
}

void displayFeatures(vector<feature> features, Mat image, string winname) {
    for(int i = 0; i < features.size(); i++) {
        Point center(features[i].x, features[i].y);

        Scalar color(0,0,255);

        circle(image, center, 4, color, 1, LINE_AA);

        Point2f dir(features[i].x + 10 * cos(features[i].angle), features[i].y + 10 * sin(features[i].angle));

        line(image, center, dir, Scalar(255,0,0), 1);
    }

    imshow(winname, image);
}

int main(int argc, char **argv) {

    


    Mat image1 = imread("../images/1.png");
    Mat image2 = imread("../images/2.png");

    vector <feature> features1 = findFeatures(image1);
    vector <feature> features2 = findFeatures(image2);

    displayFeatures(features1, image1, "1");
    displayFeatures(features2, image2, "2");

    waitKey(0);
    return 0;
}

