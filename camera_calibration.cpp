#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // ===================== USER PARAMETERS =====================
    const int numImgs = 19;
    const Size boardSize(9, 6);          // INTERNAL corners
    const float squareSize = 0.025f;     // meters (25mm)
    const string basePath = "../calib/";
    const string ext = ".jpg";
    // ============================================================

    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints;

    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            obj.push_back(Point3f(j * squareSize,
                                  i * squareSize,
                                  0));
        }
    }

    Size imageSize;

    Mat x = imread("../calib/0.jpg");
    cout << x.cols << endl << x.rows << endl;

    for (int i = 0; i < numImgs; i++) {
        string filename = basePath + to_string(i) + ext;
        Mat img = imread(filename, IMREAD_GRAYSCALE);

        if (img.empty()) {
            cerr << "Failed to load " << filename << endl;
            continue;
        }

        imageSize = img.size();

        vector<Point2f> corners;
        bool found = findChessboardCorners(
            img,
            boardSize,
            corners,
            CALIB_CB_ADAPTIVE_THRESH |
            CALIB_CB_NORMALIZE_IMAGE
        );

        if (!found) {
            cerr << "Corners not found in " << filename << endl;
            continue;
        }

        cornerSubPix(
            img,
            corners,
            Size(11, 11),
            Size(-1, -1),
            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.001)
        );

        imagePoints.push_back(corners);
        objectPoints.push_back(obj);

        cout << "Processed " << filename << endl;
    }

    if (imagePoints.size() < 10) {
        cerr << "Not enough valid images for calibration!" << endl;
        return -1;
    }

    Mat K, distCoeffs;
    vector<Mat> rvecs, tvecs;

    double rms = calibrateCamera(
        objectPoints,
        imagePoints,
        imageSize,
        K,
        distCoeffs,
        rvecs,
        tvecs
    );

    cout << "\n=== Calibration Results ===" << endl;
    cout << "RMS error: " << rms << endl;
    cout << "Camera Matrix K:\n" << K << endl;
    cout << "Distortion Coefficients:\n" << distCoeffs << endl;

    return 0;
}
