#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "ORB_pattern.h"



using namespace std;
using namespace cv;


//struct for the feature 
struct feature {
    int x;
    int y;
    double angle;
    vector<uint32_t> descriptors;
};

const double matchingRatio = 0.9;
const int max_dist = 120;

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

                                        
//gets the brightness as cols x, rows y
int getBrightness(Mat im, int x, int y) {
    uchar b = im.at<uchar>(y,x);
    int brightness = static_cast<int>(b);
    return brightness;
    
}

double getOrientation(Mat im, int x, int y) {
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

    double angle = atan2((double) m01, (double) m10); //randians
    return angle;
}

vector<uint32_t> getDescriptor( Mat gray, int x, int y, feature f)
{
    vector<uint32_t> desc(8, 0); // 8 × 32 = 256 bits

    const double c = cos(f.angle);
    const double s = sin(f.angle);

    for (int i = 0; i < 256; i++) {

     
        int px1 = ORB_pattern[4 * i];
        int py1 = ORB_pattern[4 * i + 1];
        int px2 = ORB_pattern[4 * i + 2];
        int py2 = ORB_pattern[4 * i + 3];

        // Rotate points
        int rx1 = x + cvRound(c * px1 - s * py1);
        int ry1 = y + cvRound(s * px1 + c * py1);
        int rx2 = x + cvRound(c * px2 - s * py2);
        int ry2 = y + cvRound(s * px2 + c * py2);

        // Bounds check — if ANY bit is invalid, reject descriptor
        if (rx1 < 0 || ry1 < 0 || rx1 >= gray.cols || ry1 >= gray.rows ||
            rx2 < 0 || ry2 < 0 || rx2 >= gray.cols || ry2 >= gray.rows) {
            return {}; // invalid descriptor
        }

        // Intensity comparison
        if (gray.at<uchar>(ry1, rx1) < gray.at<uchar>(ry2, rx2)) {
            desc[i >> 5] |= (1u << (i & 31));
        }
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

    //Todo: tune these
    int threshold = 13;
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
                        double angle = getOrientation(gray_image, i, j);
                        feature f;
                        f.x = i;
                        f.y = j;
                        f.angle = angle;

                        //f.descriptors = getDescriptor(gray_image, i, j, f);

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


//calculates the hamming distance between two vectors
//TODO: Optimize this for speed
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

//we basically do the reverse of matchFeatures
//we take the second feature and the entirety of the first feature list and see if the same match occurs both ways
//if so, thats good, otherwise we remove that match and dont use it
//uses same first and second as matchFeatures
bool reverseChecker(vector<feature> first, feature second, int matchedFirst) {

    int best_i = -1;
    int best = 1e9;
    int second_best = 1e9;

       
    for(int i = 0; i < first.size(); i++) {

            
        //dont need this because if it arrived at this function it is already good (hopefully) :pray

        //if (first[i].descriptors.empty() || second[j].descriptors.empty() || first[i].descriptors.size() != second[j].descriptors.size()) {
            //continue;
        //}

        int dist = hammingDistance(second.descriptors, first[i].descriptors);

        
        if(dist < best) {
            
            second_best = best;
            best = dist;
            best_i = i;

        } else if(dist < second_best) {                

            second_best = dist;

        }

            
    }

    

    if(best < max_dist && best < matchingRatio * second_best) {
        if (best_i != -1 && best < max_dist && matchedFirst == best_i) {
            return true;
        }
    }   

    return false;
    
}

vector<pair<feature, feature>> matchFeatures(vector<feature> first, vector<feature> second) {
    
    //TODO: tune these
    //double ratio = 0.9f;
    //int max_dist = 120;


    vector<pair<feature, feature>> good;

    for (int i = 0; i < first.size(); i++) {

        

        int best_j = -1;
        int best = 1e9;
        int second_best = 1e9;

       

        for(int j = 0; j < second.size(); j++) {

            

            if (first[i].descriptors.empty() || second[j].descriptors.empty() || first[i].descriptors.size() != second[j].descriptors.size()) {
                continue;
            }

            int dist = hammingDistance(first[i].descriptors, second[j].descriptors);

        
            if(dist < best) {
            
                second_best = best;
                best = dist;
                best_j = j;

            } else if(dist < second_best) {                

                second_best = dist;

            }

            
        }

    

        if(best < max_dist && best < matchingRatio * second_best) {
            if (best_j != -1 && best < max_dist && reverseChecker(first, second[best_j], i)) {
                good.push_back({first[i], second[best_j]});
            }
        }

            
        }
    

    return good;

    
}



Mat drawMatches(vector<pair<feature, feature>> matches, Mat img, int imgcols) {
    //draw the lines for matches

    Scalar color(255, 0, 0);

    for(int i = 0; i < matches.size(); i++) {
        //start point is in the left image - simple
        Point start(matches[i].first.x, matches[i].first.y);

        //end point is the x-position in the right image plus the width of the left image
        Point end(matches[i].second.x + imgcols, matches[i].second.y);
        line(img, start, end, color, 1);
    }

    return img;

}

//TODO: wrap all of this in a function that takes in the images as arguements

int main(int argc, char **argv) {

    Mat image1 = imread("../images/4.png");
    Mat image2 = imread("../images/3.png");

    const int goalcols = 640;

    double fx1 = (double) goalcols / image1.cols;
    double fy1 = fx1;

    double fx2 = (double) goalcols / image1.cols;
    double fy2 = (image1.rows * fy1) / image2.rows;


    resize(image1, image1, Size(), fx1, fy1, INTER_AREA);
    resize(image2, image2, Size(), fx2, fy2, INTER_AREA);

    cout << "The dimensions of the first image are: " << image1.cols << " x " << image1.rows << endl;
    cout << "The dimensions of the second image are: " << image2.cols << " x " << image2.rows << endl;

    Mat gray_image_1;
    Mat gray_image_2;
    
    cvtColor(image1, gray_image_1, COLOR_BGR2GRAY);
    cvtColor(image2, gray_image_2, COLOR_BGR2GRAY);

    vector <feature> features1 = findFeatures(image1);
    vector <feature> features2 = findFeatures(image2);

    cout << "The number of features of the first image is: " << features1.size() << endl;
    cout << "The number of features of the second image is: " << features2.size() << endl;

    for(int i = 0; i < features1.size(); i++) {
        features1[i].descriptors = getDescriptor(gray_image_1, features1[i].x, features1[i].y, features1[i]);
    } 

    //deletes features with empty descriptors
    features1.erase(remove_if(features1.begin(), features1.end(), [](const feature& f) {return f.descriptors.size() != 8;}),features1.end());

    for(int i = 0; i < features2.size(); i++) {
        features2[i].descriptors = getDescriptor(gray_image_2, features2[i].x, features2[i].y, features2[i]);
    }

    features2.erase(remove_if(features2.begin(), features2.end(), [](const feature& f) {return f.descriptors.size() != 8;}),features2.end());

    vector<pair<feature, feature>> matches = matchFeatures(features1, features2);

    cout << "The number of matches is: " << matches.size() << endl;

    image1 = drawFeatures(features1, image1);
    image2 = drawFeatures(features2, image2);

    Mat combined;

    //image 1 is on the left
    hconcat(image1, image2, combined);    

    combined = drawMatches(matches, combined, image2.cols);

    imshow("s", combined);

    waitKey(0);

    return 0;
}

