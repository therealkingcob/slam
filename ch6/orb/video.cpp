#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

int main() {

    // Open the camera
    VideoCapture cap("../test1.mp4");


    if (!cap.isOpened()) {
        cout << "Cannot open camera!" << endl;
        return -1;
    }

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    Mat prev_frame;
    vector<KeyPoint> prev_keypoints;
    Mat prev_descriptors;

    // Grab first frame so we have a "previous" to match against
    cap >> prev_frame;
    if (prev_frame.empty()) return -1;
    resize(prev_frame, prev_frame, Size(), 0.4, 0.4);

    detector->detect(prev_frame, prev_keypoints);
    descriptor->compute(prev_frame, prev_keypoints, prev_descriptors);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(), 0.4, 0.4);

        // ORB for current frame
        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detect(frame, keypoints);
        descriptor->compute(frame, keypoints, descriptors);

        if (!descriptors.empty() && !prev_descriptors.empty()) {

            // Match with previous frame
            vector<DMatch> matches;
            matcher->match(prev_descriptors, descriptors, matches);

            // Filter matches
            double min_dist = 10000;
            for (auto &m : matches)
                if (m.distance < min_dist) min_dist = m.distance;

            vector<DMatch> good_matches;
            for (auto &m : matches)
                if (m.distance <= max(2 * min_dist, 30.0))
                    good_matches.push_back(m);

            // Draw matches
            Mat img_match;
            drawMatches(prev_frame, prev_keypoints, frame, keypoints,
                        good_matches, img_match);

            imshow("Frame-to-Frame ORB Tracking", img_match);
        }

        // Update previous frame
        prev_frame = frame.clone();
        prev_keypoints = keypoints;
        prev_descriptors = descriptors.clone();

        // Exit on ESC
        if (waitKey(1) == 27) break;
    }

    return 0;
}