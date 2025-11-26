#include <iostream> 

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include <chrono>

using namespace std;
using namespace cv;


int main(int argc, char **argv) {
    /*
    if(argc != 3) {
        cout << "usage: feature extraction img1 img2" << endl;
        return 1;
    }

    */

    //read images

    Mat img_1 = imread("../images_1/3.png", IMREAD_COLOR);
    Mat img_2 = imread("../images_1/4.png", IMREAD_COLOR);


    resize(img_1, img_1, Size(), 0.1, 0.1); 
    resize(img_2, img_2, Size(), 0.1, 0.1);

    //init

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);


    descriptor->compute(img_1,keypoints_1,descriptors_1);
    descriptor->compute(img_2,keypoints_2,descriptors_2);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    Mat outimg1;
    
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("Orb Features", outimg1);

    vector<DMatch> matches;
    
    t1 = chrono::steady_clock::now();

    matcher->match(descriptors_1, descriptors_2, matches);

    t2 = chrono::steady_clock::now();

    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "match ORB cost = " << time_used.count() << " seconds." << endl; 

    auto min_max = minmax_element(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2) {return m1.distance < m2.distance;});

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max Dist : %f \n", max_dist);
    printf("-- Min Dist : %f \n", min_dist);

    vector<DMatch> good_matches;

    for(int i = 0; i < descriptors_1.rows; i++) {
        if(matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;

    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey(0);

    return 0;

}