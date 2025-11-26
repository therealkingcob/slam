#include <opencv2/opencv.hpp>
#include <string>
#include <nmmintrin.h>
#include <chrono>

using namespace std;

//the file path of both images
string first_file = "./1.png";
string second_file = "./2.png";

//32 bit unsigned vector 
typedef vector<uint32_t> DescType;



void computeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors);




void BFMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches);


int main(int argc, char **argv) {

    //load image

    cv::Mat first_image = cv::imread(first_file, 0);

    cv::Mat second_image = cv::imread(second_file, 0);

    //assert that both exist
    assert(first_image.data != nullptr && second_image.data != nullptr);

    //start the timer at the current time

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    //detect keypoints for image 1
    vector<cv::KeyPoint> keypoints1;
    cv::FAST(first_image, keypoints1, 40);
    vector<DescType> descriptor1;
    computeORB(first_image, keypoints1, descriptor1);

    //detect keypoints for image 2

    vector<cv::KeyPoint> keypoints2;
    cv::FAST(second_image, keypoints2, 40);
    vector<DescType> descriptor2;
    computeORB(second_image, keypoints2, descriptor2);
    
    //get the second time point for the timer

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    //total time used

    chrono::duration <double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);

    cout << "Extract ORB cost: " << time_used.count() << " seconds" << endl;




}


void computeORB(const cv::Mat &img, vector <cv::KeyPoint> &keypoints, vector<DescType> &descriptors) {

    const int half_patch_size = 8;
    const int half_boundary = 16;
    int bad_points = 0;


    //for every keypoint
    for(auto &kp: keypoints) {
        //bad points
        if(kp.pt.x < half_boundary || kp.pt.y < half_boundary || kp.pt.x >= img.cols - half_boundary || kp.pt.y >= img.rows - half_boundary) {
            bad_points++;
            descriptors.push_back({});
            continue;
        }

        float m01 = 0, m10 = 0;

        for(int dx = -half_patch_size; dx )
    }
}