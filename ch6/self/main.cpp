#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "ORB_pattern.h"



using namespace std;
using namespace cv;

struct feature {
    int x;
    int y;
    float angle;
    vector<uint32_t> descriptors;
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

            int dx = xx - x;
            int dy = yy - y;

            m10 += dx * I;
            m01 += dy * I;

        }
    }

    float angle = atan2((float) m01, (float) m10); //randians
    return angle;
}

vector<uint32_t> getDescriptor(Mat image, int x, int y, feature f) {
    vector<uint32_t> desc;

    //Mat gray_image;
    

    for(int i = 0; i < 8; i++) {
        uint32_t d =0;
        for(int j = 0; j < 32; j++) {

            int index = i * 32 + j;

            int px1 = ORB_pattern[index * 4];
            int py1 = ORB_pattern[index * 4 + 1];
            int px2 = ORB_pattern[index * 4 + 2];
            int py2 = ORB_pattern[index * 4 + 3];



            float rx1 = cos(f.angle) * px1 - sin(f.angle) * py1;
            float ry1 = sin(f.angle) * px1 + cos(f.angle) * py1;


            float rx2 = cos(f.angle) * px2 - sin(f.angle) * py2;
            float ry2 = sin(f.angle) * px2 + cos(f.angle) * py2;

            int xx1 = x + (int)round(rx1);
            int yy1 = y + (int)round(ry1);

            if(xx1 < 0 || yy1 < 0 || xx1 >= image.cols || yy1 >= image.rows) {continue;}




            if(image.at<uchar> (y + (int) round(ry1), x + (int) round(rx1)) < image.at<uchar> (y + (int) round(ry2), x + (int) round(rx2))) {
                d |= 1 << j;
            }

        }
        desc.push_back(d);
    }

    return desc;


}

vector<feature> findFeatures(Mat image) {
    int width = image.cols;
    int height = image.rows;

    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    GaussianBlur(gray_image, gray_image, Size(5,5), 1.0);


    vector <feature> features;

    int threshold = 12;
    int tolerance = 30;

    //i is the x-axis
    //j is the y-axis

    for (int i = 4; i < width-4; i++) {
        for(int j = 4; j < height-4; j++) {

            int b = getBrightness(gray_image, i, j);

            int brighter = 0;
            int darker = 0;

            //we need to go 16 pixels for the entire radius plus n since the series to feature pixel candidates could start on the last pixel of the 16
            for(int f = 0; f < 16 + threshold; f++) {

                int brightnessAtPoint = getBrightness(gray_image, i + offsets[f][0], j - offsets[f][1]);

                if(brightnessAtPoint > b + tolerance) {

                    brighter++;
                    darker = 0;

                } else if (brightnessAtPoint < b - tolerance){
                    
                    darker++;
                    brighter = 0;

                } else {
                    darker = 0;
                    brighter = 0;
                }

                if(brighter >= threshold || darker >= threshold) {
                        float angle = getOrientation(gray_image, i, j);
                        feature f;
                        f.x = i;
                        f.y = j;
                        f.angle = angle;

                        f.descriptors = getDescriptor(gray_image, i, j, f);

                        features.push_back(f);

                        //cout << "inside second if statement" << endl;
                        break;
                    }
                    
            }

            //cout << "after the second loop" << endl;
        }
    }

    return features;
}

Mat drawFeatures(vector<feature> features, Mat image) {
    for(int i = 0; i < features.size(); i++) {
        Point center(features[i].x, features[i].y);

        Scalar color(0,0,255);

        circle(image, center, 4, color, 1, LINE_AA);

        Point2f dir(features[i].x + 10 * cos(features[i].angle), features[i].y + 10 * sin(features[i].angle));

        line(image, center, dir, Scalar(255,0,0), 1);
    }

    return image;
}






int hammingDistance(vector<uint32_t>a, vector<uint32_t>b) {

    int dist = 0;

    for(int i = 0; i < 8; i++) {
        uint32_t v = a[i] ^ b[i];

        int bits = 0;
        while(v) {
            bits += v & 1;
            v>>= 1;
        }

        dist += bits;
    }

    return dist;

}

vector<pair<feature, feature>> matchFeatures(vector<feature> first, vector<feature> second) {
    //our max hamming distance is 40

    int max_dist = 50;

    vector<pair<feature, feature>> good;

    for (int i = 0; i < first.size(); i++) {
        int best_j = -1;
        int best_d = 1e9;

        for(int j = 0; j < second.size(); j++) {
            int dist = hammingDistance(first[i].descriptors, second[j].descriptors);

            if(dist < best_d) {
                best_d = dist;
                best_j = j;

            }
        }

        if(best_d < max_dist) {
            good.push_back({first[i], second[best_j]});
        }
    }

    return good;
}

Mat drawMatches(vector<pair<feature, feature>> matches, Mat img) {
    //draw the lines for matches

    Scalar color(255, 0, 0);

    for(int i = 0; i < matches.size(); i++) {
        Point start(matches[i].first.x, matches[i].first.y);
        Point end(matches[i].second.x + img.cols/2, matches[i].second.y);
        line(img, start, end, color, 2);
    }

    return img;

    
    

}

int main(int argc, char **argv) {

    Mat image1 = imread("../images/1.png");
    Mat image2 = imread("../images/2.png");

    vector <feature> features1 = findFeatures(image1);
    vector <feature> features2 = findFeatures(image2);

    image1 = drawFeatures(features1, image1);
    image2 = drawFeatures(features2, image2);

    Mat combined;

    hconcat(image2, image1, combined);

    

    //float distance = getDistance(image1, image2, 10, 10, 10, 10);

    //cout << distance << " Number of rows is: " << image1.rows << endl; 

    //vector<vector<double>> distances;

    Mat gray_image_1;
    Mat gray_image_2;
    cvtColor(image1, gray_image_1, COLOR_BGR2GRAY);
    cvtColor(image2, gray_image_2, COLOR_BGR2GRAY);


    for(int i = 0; i < features1.size(); i++) {
        features1[i].descriptors = getDescriptor(gray_image_1, features1[i].x, features1[i].y, features1[i]);
    } 

    for(int i = 0; i < features2.size(); i++) {
        features2[i].descriptors = getDescriptor(gray_image_2, features2[i].x, features2[i].y, features2[i]);
    }

    vector<pair<feature, feature>> matches = matchFeatures(features1, features2);

    combined = drawMatches(matches, combined);

    imshow("s", combined);

    waitKey(0);

    return 0;
}

