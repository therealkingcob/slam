#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

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

const int goalcols = 640;

const vector <vector<int>> offsets = {{0 ,4},
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
int getBrightness(const Mat& im, int x, int y) {
    uchar b = im.at<uchar>(y,x);
    int brightness = static_cast<int>(b);
    return brightness;
    
}

double getOrientation(const Mat& im, int x, int y) {
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

vector<uint32_t> getDescriptor(const Mat& gray, int x, int y, feature f)
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

//to access a certain pixels brightness, we go b[y*img.cols+x]
vector<int> imgBrightness(const Mat& img) {
    int x = img.cols;
    int y = img.rows;

    vector<int> b(x * y);

    //fits goes (0,0), (0,1), (0,2) ... (1,0), (1,1) ...

    for(int i = 0; i < y; i++) {

        const uchar* row = img.ptr<uchar>(i);

        for(int j = 0; j < x; j++) {
            
            b[i * x + j] = row[j];

        }
    }

    return b;
    
}


vector<feature> findFeatures(const Mat& gray_image, const vector<int>& b_img) {
    int width = gray_image.cols;
    int height = gray_image.rows;

    //Mat gray_image;
    //cvtColor(image, gray_image, COLOR_BGR2GRAY);

    GaussianBlur(gray_image, gray_image, Size(5,5), 1.0);


    vector <feature> features;

    //Todo: tune these
    int threshold = 15;
    int tolerance = 40; //the bigger this is, the less features there are

    //i is the x-axis
    //j is the y-axis

    for (int i = 4; i < width-4; i++) {
        for(int j = 4; j < height-4; j++) {

            int b = b_img[j*width + i];
            //b[y*img.cols+x]

            int brighter = 0;
            int darker = 0;

            //we need to go 16 pixels for the entire radius plus n since the series to feature pixel candidates could start on the last pixel of the 16
            for(int f = 0; f < 16 + threshold; f++) {

                //int brightnessAtPoint = getBrightness(gray_image, i + offsets[f][0], j - offsets[f][1]);
                int brightnessAtPoint = b_img[((j - offsets[f][1]) * width) + (i + offsets[f][0])];

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


Mat drawFeatures(const vector<feature>& features, const Mat& image) {
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
int hammingDistance(const vector<uint32_t>& a, const vector<uint32_t>& b) {

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
bool reverseChecker(const vector<feature>& first, const feature& second, int matchedFirst) {

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

vector<pair<feature, feature>> matchFeatures(const vector<feature>& first, const vector<feature>& second) {
    
    //TODO: tune these
    //double ratio = 0.9f;
    //int max_dist = 120;


    vector<pair<feature, feature>> good;

    for (int i = 0; i < first.size(); i++) {

        

        int best_j = -1;
        int best = 1e9;
        int second_best = 1e9;

       

        for(int j = 0; j < second.size(); j++) {

            

            //if (first[i].descriptors.empty() || second[j].descriptors.empty() || first[i].descriptors.size() != second[j].descriptors.size()) {
              //  continue;
            //}

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



Mat drawMatches(const vector<pair<feature, feature>>& matches, const Mat& img, int imgcols) {
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

    chrono::steady_clock::time_point m1 = chrono::steady_clock::now();

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    Mat image1 = imread("../images/4.png", IMREAD_GRAYSCALE);
    Mat image2 = imread("../images/3.png", IMREAD_GRAYSCALE);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to read images is: " << time_used.count() << endl;

    t1 = chrono::steady_clock::now();

    double fx1 = (double) goalcols / image1.cols;
    double fy1 = fx1;

    double fx2 = (double) goalcols / image1.cols;
    double fy2 = (image1.rows * fy1) / image2.rows;

    resize(image1, image1, Size(), fx1, fy1, INTER_AREA);
    resize(image2, image2, Size(), fx2, fy2, INTER_AREA);

    t2 = chrono::steady_clock::now();

    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to scale images is: " << time_used.count() << endl;


    cout << "The dimensions of the first image are: " << image1.cols << " x " << image1.rows << endl;
    cout << "The dimensions of the second image are: " << image2.cols << " x " << image2.rows << endl;

    t1 = chrono::steady_clock::now();

    //Mat gray_image1;
    //Mat gray_image2;
    
    //cvtColor(image1, gray_image1, COLOR_BGR2GRAY);
    //cvtColor(image2, gray_image2, COLOR_BGR2GRAY);

    t2 = chrono::steady_clock::now();

    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to make images gray is: " << time_used.count() << endl;

    t1 = chrono::steady_clock::now();

    vector<int> gray_image1_b = imgBrightness(image1);
    vector<int> gray_image2_b = imgBrightness(image2);

    t2 = chrono::steady_clock::now();

    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to get the entire brightness is: " << time_used.count() << endl;

    t1 = chrono::steady_clock::now();

    vector <feature> features1 = findFeatures(image1, gray_image1_b);
    vector <feature> features2 = findFeatures(image2, gray_image2_b);

     t2 = chrono::steady_clock::now();

    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to find features is: " << time_used.count() << endl;

    cout << "The number of features of the first image is: " << features1.size() << endl;
    cout << "The number of features of the second image is: " << features2.size() << endl;

     t1 = chrono::steady_clock::now();

    for(int i = 0; i < features1.size(); i++) {
        features1[i].descriptors = getDescriptor(image1, features1[i].x, features1[i].y, features1[i]);
    } 

    for(int i = 0; i < features2.size(); i++) {
        features2[i].descriptors = getDescriptor(image2, features2[i].x, features2[i].y, features2[i]);
    }

     t2 = chrono::steady_clock::now();

    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to find descriptors is: " << time_used.count() << endl;

     t1 = chrono::steady_clock::now();

    //deletes features with empty descriptors
    features1.erase(remove_if(features1.begin(), features1.end(), [](const feature& f) {return f.descriptors.size() != 8;}),features1.end());
    features2.erase(remove_if(features2.begin(), features2.end(), [](const feature& f) {return f.descriptors.size() != 8;}),features2.end());

     t2 = chrono::steady_clock::now();

     time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to remove bad descriptors is: " << time_used.count() << endl;



     t1 = chrono::steady_clock::now();


    vector<pair<feature, feature>> matches = matchFeatures(features1, features2);

     t2 = chrono::steady_clock::now();

     time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to match features is: " << time_used.count() << endl;

    cout << "The number of matches is: " << matches.size() << endl;

     t1 = chrono::steady_clock::now();

    image1 = drawFeatures(features1, image1);
    image2 = drawFeatures(features2, image2);

     t2 = chrono::steady_clock::now();

     time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to draw features is: " << time_used.count() << endl;

    Mat combined;

    //image 1 is on the left
    hconcat(image1, image2, combined);    

     t1 = chrono::steady_clock::now();

    combined = drawMatches(matches, combined, image2.cols);

     t2 = chrono::steady_clock::now();

     time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Time to draw matches is: " << time_used.count() << endl;

    chrono::steady_clock::time_point m2 = chrono::steady_clock::now();

    chrono::duration<double> mtime_used = chrono::duration_cast<chrono::duration<double>>(m2-m1);

    cout << "Total time is: " << mtime_used.count() << endl;

    imshow("s", combined);

    waitKey(0);

    return 0;
}

