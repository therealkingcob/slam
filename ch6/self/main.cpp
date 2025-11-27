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

void displayFeatures(vector<feature> features, Mat image) {
    for(int i = 0; i < features.size(); i++) {
        Point center(features[i].x, features[i].y);

        Scalar color(0,0,255);

        circle(image, center, 4, color, 1, LINE_AA);

        Point2f dir(features[i].x + 10 * cos(features[i].angle), features[i].y + 10 * sin(features[1].angle));

    line(image, center, dir, Scalar(255,0,0), 1);
    }

    imshow("Image with Circles", image);
}

int main(int argc, char **argv) {

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


    Mat image = imread("../images/1.png");


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

    displayFeatures(features, image);

    waitKey(0);
    return 0;
}

/*
Select pixel p in the image, assuming its brightness as Ip
2. Set a threshold T (for example, 20% of Ip).
2But now we have many cheap embedded GPU chips, so it may not be an issue anymore.
6.1. FEATURE METHOD 131
3. Take the pixel p as the center, and select the 16 pixels on a circle with a radius
of 3.
4. If there are consecutive N points on the selected circle whose brightness is
greater than Ip+T or less than Ip−T, then the central pixel p can be considered
a feature point. N usually takes 12, which is FAST-12. Other commonly used
N values are 9 and 11, called FAST-9 and FAST-11, respectively).
5. Iterate through the above four steps on each pixel.
*/
