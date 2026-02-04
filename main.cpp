#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>
#include <fstream>
#include <ceres/ceres.h>


#include "ORB_pattern.h"
//#include "feature.h"

using namespace std;
using namespace cv;

/*

//bundle adjustment stuff
struct TrackedFeature {
    int track_id;
    cv::Point2f pt;
};

struct Landmark {
    int id;
    double xyz[3];
};

struct Observation {
    int keyframe_id;
    int landmark_id;
    cv::Point2f pixel;
};

struct ReprojectionError {
    ReprojectionError(double u, double v, const cv::Mat& K)
        : u_(u), v_(v),
          fx_(K.at<double>(0,0)),
          fy_(K.at<double>(1,1)),
          cx_(K.at<double>(0,2)),
          cy_(K.at<double>(1,2)) {}

    template <typename T>
    bool operator()(const T* const pose,
                    const T* const point,
                    T* residuals) const {

        // pose: [angle-axis (3), translation (3)]
        T p[3];
        ceres::AngleAxisRotatePoint(pose, point, p);

        p[0] += pose[3];
        p[1] += pose[4];
        p[2] += pose[5];

        // Reject behind camera implicitly via residual explosion
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T u = T(fx_) * xp + T(cx_);
        T v = T(fy_) * yp + T(cy_);

        residuals[0] = u - T(u_);
        residuals[1] = v - T(v_);

        return true;
    }

    static ceres::CostFunction* Create(
        double u, double v, const cv::Mat& K) {

        return new ceres::AutoDiffCostFunction<
            ReprojectionError, 2, 6, 3>(
            new ReprojectionError(u, v, K));
    }

    double u_, v_;
    double fx_, fy_, cx_, cy_;
};


*/

double scale  = 50;

    
const double matchingRatio = 0.95;

const int max_dist = 400;

const int goalcols = 1280;

vector <double> global_position = {0, 0, 0}; //x,y,z

struct feature {
    int x;
    int y;
    double angle;
    std::vector <uint32_t> descriptors;
};

int featureMax = 100;

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


                                           /*
void runBundleAdjustment(
    std::vector<Keyframe>& keyframes,
    std::vector<Landmark>& landmarks,
    const std::vector<Observation>& observations,
    const cv::Mat& K
) {
    ceres::Problem problem;

    // Add residuals
    for (const auto& obs : observations) {
        Keyframe& kf = keyframes[obs.keyframe_id];
        Landmark& lm = landmarks[obs.landmark_id];

        ceres::CostFunction* cost =
            ReprojectionError::Create(
                obs.pixel.x, obs.pixel.y, K);

        problem.AddResidualBlock(
            cost,
            new ceres::HuberLoss(1.0), // important
            kf.pose,
            lm.xyz
        );
    }

    // Gauge fixing: fix first keyframe
    problem.SetParameterBlockConstant(keyframes[0].pose);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 15;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

*/
                                        
//gets the brightness as cols x, rows y
int getBrightness(const Mat& im, int x, int y) {
    uchar b = im.at<uchar>(y,x);
    int brightness = static_cast<int>(b);
    return brightness;
    
}

void savePointCloudPLY(
    const std::string& filename,
    const std::vector<cv::Point3d>& points
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }

    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";

    for (const auto& p : points) {
        file << p.x << " " << p.y << " " << p.z << "\n";
        //cout << "Writing point: " << p.x << " " << p.y << " " << p.z << endl;
    }

    file.close();
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

    const double c = 1.0;
    const double s = 0.0;

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
            continue; // invalid descriptor
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

    


    vector <feature> features;

    //Todo: tune these
    int threshold = 12;
    int tolerance = 30; //the bigger this is, the less features there are

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

vector<pair<feature, feature>> featureReadingAndMatching(Mat img1, Mat img2) {
    //Mat img1 = imread(img1, IMREAD_GRAYSCALE);
    //Mat img2 = imread(img2, IMREAD_GRAYSCALE);

    cout << "matching" << endl;


    double fx1 = (double) goalcols / img1.cols;
    double fy1 = fx1;

    double fx2 = (double) goalcols / img1.cols;
    double fy2 = (img1.rows * fy1) / img2.rows;

 
    if(img1.channels() == 3) cvtColor(img1, img1, COLOR_BGR2GRAY);

    if(img2.channels() == 3) cvtColor(img2, img2, COLOR_BGR2GRAY);


    GaussianBlur(img1, img1, Size(5,5), 1.0);
    GaussianBlur(img2, img2, Size(5,5), 1.0);

    Mat img1_small, img2_small;
    resize(img1, img1_small, Size(), 0.5, 0.5, INTER_LINEAR);
    resize(img2, img2_small, Size(), 0.5, 0.5, INTER_LINEAR);
    
    // Ensure matrices are continuous in memory
    if (!img1.isContinuous()) img1 = img1.clone();
    if (!img2.isContinuous()) img2 = img2.clone();



    vector<int> b1 = imgBrightness(img1);
    vector<int> b1s = imgBrightness(img1_small);

    vector<int> b2 = imgBrightness(img2);
    vector<int> b2s = imgBrightness(img2_small);



    vector<feature> f1 = findFeatures(img1, b1);
    vector<feature> f1s = findFeatures(img1_small, b1s);

    vector<feature> f2 = findFeatures(img2, b2);
    vector<feature> f2s = findFeatures(img2_small, b2s);

    for (auto& f : f1s) {
        f.x *= 2;
        f.y *= 2;
    }
    for (auto& f : f2s) {
        f.x *= 2;
        f.y *= 2;
    }

    f1.insert(f1.end(), f1s.begin(), f1s.end());
    f2.insert(f2.end(), f2s.begin(), f2s.end());


    for(int i = 0; i < f1.size(); i++) {
        f1[i].descriptors = getDescriptor(img1, f1[i].x, f1[i].y, f1[i]);
    } 

    for(int i = 0; i < f2.size(); i++) {
        f2[i].descriptors = getDescriptor(img2, f2[i].x, f2[i].y, f2[i]);
    }


    //deletes features with empty descriptors
    f1.erase(remove_if(f1.begin(), f1.end(), [](const feature& f) {return f.descriptors.size() != 8;}),f1.end());
    f2.erase(remove_if(f2.begin(), f2.end(), [](const feature& f) {return f.descriptors.size() != 8;}),f2.end());


    vector<pair<feature, feature>> matches = matchFeatures(f1, f2);


    //img1 = drawFeatures(features1, img1);
    //img2 = drawFeatures(features2, img2);

/*
    Mat combined;

    //image 1 is on the left
    hconcat(img1, img2, combined);    


    combined = drawMatches(matches, combined, img2.cols);

    

    //imshow("s", combined);

    //waitKey(1) == 27;
*/
    return matches;

}

/*

point normalizePoint(pair<int, int> coords, Mat k) {
    k = k.inv();
    Mat pt = (Mat_<double>(3,1) << p.first, p.second, 1.0);
    Mat pn = K * pt;

    return point(pn.at<double>(0), pn.at<double>(1), 1.0);
}

//Mat essentialMatrix(vector<pair<feature,feature>> matches, Mat k) {
//    int n = matches.size();
//    Mat kinv = k.inv(); 

//}

*/

vector<cv::Point3d> triangulatePointsBetweenFrames(
    const vector<Point2f>& points1,
    const vector<Point2f>& points2,
    const Mat& K,
    const Mat& R1, const Mat& t1,
    const Mat& R2, const Mat& t2
) {

    vector<cv::Point3d> points3D;

    if (points1.size() < 20) return points3D;
    if (points1.size() != points2.size()) return points3D;
    if (norm(t1-t2) < 1e-4) return points3D;

    Mat P1, P2;
    hconcat(R1, t1, P1);
    hconcat(R2, t2, P2);
    P1 = K * P1;
    P2 = K * P2;

    Mat points4D;
    triangulatePoints(P1, P2, points1, points2, points4D);
    points4D.convertTo(points4D, CV_64F);

    //vector<cv::Point3d> points3D;
    points3D.reserve(points4D.cols);

    for (int i = 0; i < points4D.cols; i++) {
        double w = points4D.at<double>(3, i);
        if (fabs(w) < 1e-6) continue;

        cv::Point3d p(
            points4D.at<double>(0, i) / w,
            points4D.at<double>(1, i) / w,
            points4D.at<double>(2, i) / w
        );

        // Reject behind-camera or insane depth
        if (p.z <= 0 || p.z > 1000.0) continue;

        points3D.push_back(p);
    }
    return points3D;
}

double reprojectionError(
    const cv::Point3d& X,
    const cv::Point2f& x,
    const Mat& K,
    const Mat& R,
    const Mat& t
) {
    Mat Xc = R * (Mat_<double>(3,1) << X.x, X.y, X.z) + t;
    if (Xc.at<double>(2) <= 0) return 1e9;

    Mat xp = K * Xc;
    double u = xp.at<double>(0) / xp.at<double>(2);
    double v = xp.at<double>(1) / xp.at<double>(2);

    return hypot(u - x.x, v - x.y);
}

void drawImage(Mat a) {
    //destroyAllWindows();
    imshow("a", a);
    waitKey(1);
}

void pose_estimation(vector<pair<feature,feature>> matches, Mat& pos, Mat& rot) {
    //our intrinsics
    Mat K = (Mat_<double>(3, 3) << 983.5778865703971, 0, 656.3414928103965, 0, 987.4489280701677, 381.2353225388408, 0, 0, 1);

    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i = 0; i < matches.size(); i++) {

        points1.push_back(Point2f(matches[i].first.x, matches[i].first.y));
        points2.push_back(Point2f(matches[i].second.x, matches[i].second.y));
    }

    Point2d principal_point(656.3414928103965, 381.2353225388408);
    double focal_length = 987.4489280701677;

    Mat essential_matrix;
    
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    Mat R,t;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);

    rot = R;
    pos = t;
}

void pose_estimation1(vector<Point2f> good_prev, vector<Point2f> good_curr, Mat& pos, Mat& rot) {
    //our intrinsics
    Mat K = (Mat_<double>(3, 3) << 983.5778865703971, 0, 656.3414928103965, 0, 987.4489280701677, 381.2353225388408, 0, 0, 1);

    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i = 0; i < good_prev.size(); i++) {
        points1.push_back(good_prev[i]);
        points2.push_back(good_curr[i]);
    }

    Point2d principal_point(656.3414928103965, 381.2353225388408);
    double focal_length = 987.4489280701677;

    Mat essential_matrix;
    
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    Mat R,t;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);

    rot = R;
    pos = t;
}

vector<double> convertAngles(Mat R) {
    double roll, pitch, yaw;

    pitch = asin(-R.at<double>(2, 0));

    if (cos(pitch) > 1e-6) {
        roll = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        yaw  = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        // Gimbal lock
        roll = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        yaw  = 0;
    }

    roll  *= 180.0 / CV_PI;
    pitch *= 180.0 / CV_PI;
    yaw   *= 180.0 / CV_PI;

    vector<double> angles;
    angles = {roll, pitch, yaw};
    return angles;
}

//expects x and y coordinates between -1 and 1
//new x = x/z
//new y = y/z

// our coords are between 
void visualizer(Mat& img, double x, double y) {

    //first point is top left
    //second point is bottom right
    int width = 1280;
    int height = 720;
    
    Point2f world(x,y);
    Point2f cam_center(0.0,0.0);

    Point2f rel = world - cam_center;

    int xx= static_cast<int>(rel.x * scale + width / 2);
    int yy = static_cast<int>(-rel.y * scale + height / 2); // 
    cout << xx << "              " << yy << endl;

    if(xx < 0 || xx > 1280 || yy < 0 || yy > 720) {
        scale *= 0.5;
    }

    cv::circle(img, {xx, yy}, 5, {0, 255, 0}, 1);


}
    

int main(int argc, char **argv) {

    double total_times;

    chrono::steady_clock::time_point time_start;
    chrono::steady_clock::time_point time_end;
    chrono::duration<double> time_used;

    Mat R_global = Mat::eye(3,3,CV_64F);
    Mat t_global = Mat::zeros(3,1,CV_64F);

    

    VideoCapture cap("../test.mp4");

    if (!cap.isOpened()) {
        cerr << "ERROR: Could not open video file\n";
        return -1;
    }

    Mat prev_frame, curr_frame;
    cap >> prev_frame;

    cout << "prev_frame size: " << prev_frame.size() << endl;

    int frame_count = 0;
    int skip_frames = 0;

    vector<pair<feature, feature>> matches;

    bool first_frame = true;

    vector<Point3d> global_map;

    while (cap.read(curr_frame)) {
        time_start = chrono::steady_clock::now();

        frame_count++;

        Mat R, t;

        if(prev_frame.channels() == 3)cvtColor(prev_frame, prev_frame, COLOR_BGR2GRAY);

        if(curr_frame.channels() == 3) cvtColor(curr_frame, curr_frame, COLOR_BGR2GRAY);

        if (first_frame) {
            matches = featureReadingAndMatching(prev_frame, curr_frame);
            first_frame = false;
        }

        vector<Point2f> good_prev, good_curr;

        // Track features using optical flow
        if (!matches.empty()) {
            vector<Point2f> tracked_pts;
            for (auto& m : matches) tracked_pts.push_back(Point2f(m.second.x, m.second.y));

            vector<Point2f> next_pts;
            vector<uchar> status;
            vector<float> err;

            calcOpticalFlowPyrLK(prev_frame, curr_frame,
                                tracked_pts, next_pts,
                                status, err);

            for (int i = 0; i < status.size(); i++) {
                if (status[i]) {
                    good_prev.push_back(tracked_pts[i]);
                    good_curr.push_back(next_pts[i]);
                }
            }
        }

        // Only estimate pose if we have enough tracked points
        if (good_prev.size() >= 8 && good_curr.size() >= 8) {
            pose_estimation1(good_prev, good_curr, t, R);

            if (!R.empty() && !t.empty()) {
                t.convertTo(t, CV_64F);
                t_global += R_global * t;
                R_global = R * R_global;
            } else {
                cout << "Warning: R or t empty, skipping global pose update\n";
                skip_frames++;
            }
        } else {
            cout << "Warning: insufficient points for pose estimation\n";
            skip_frames++;
        }

        

        // Build point correspondences
        vector<Point2f> pts1, pts2;
        for (int i = 0; i < good_prev.size(); i++) {
            pts1.push_back(good_prev[i]);
            pts2.push_back(good_curr[i]);
        }

    // Intrinsics
        Mat K = (Mat_<double>(3,3) << 983.5778865703971, 0, 656.3414928103965, 0, 987.4489280701677, 381.2353225388408, 0, 0, 1);

    // Camera poses
        Mat R1 = Mat::eye(3,3,CV_64F);
        Mat t1 = Mat::zeros(3,1,CV_64F);
        Mat R2 = R;
        Mat t2 = t;

    // Triangulate

        vector<Point3d> pts3D = triangulatePointsBetweenFrames(pts1, pts2, K, R1, t1, R2, t2);

        vector<cv::Point3d> filtered;
        for (int i = 0; i < pts3D.size(); i++) {
            double err1 = reprojectionError(pts3D[i], pts1[i], K, R1, t1);
            double err2 = reprojectionError(pts3D[i], pts2[i], K, R2, t2);

            if (err1 < 2.0 && err2 < 2.0)
                filtered.push_back(pts3D[i]);
        }
        pts3D.swap(filtered);

        for (auto& p : pts3D) {
            Mat X = (Mat_<double>(3,1) << p.x, p.y, p.z);
            Mat Xw = R_global * X + t_global;
            global_map.emplace_back(
                Xw.at<double>(0),
                Xw.at<double>(1),
                Xw.at<double>(2)
            );
        }

        if (frame_count % 20 == 0 && global_map.size() > 100) {
            savePointCloudPLY("map.ply", global_map);
            }

        time_end = chrono::steady_clock::now();

        time_used = chrono::duration_cast<chrono::duration<double>>(time_end-time_start);

        total_times += time_used.count();

        // Display info
        putText(curr_frame, "Frame: " + to_string(frame_count), Point(30,30), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
        putText(curr_frame, "Frames skipped: " + to_string(skip_frames), Point(30,60), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
        putText(curr_frame, "Position: x=" + to_string(t_global.at<double>(0)) +
                            " y=" + to_string(t_global.at<double>(1)) +
                            " z=" + to_string(t_global.at<double>(2)),
                            Point(30,90), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
        putText(curr_frame, "Orientation: roll=" + to_string(convertAngles(R_global)[0]) +
                            " pitch=" + to_string(convertAngles(R_global)[1]) +
                            " yaw=" + to_string(convertAngles(R_global)[2]),
                            Point(30,120), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
        putText(curr_frame, "Tracked points: " + to_string(good_prev.size()), Point(30,150), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);
        putText(curr_frame, "Time per frame: " + to_string(time_used.count() * 1000) + "ms", Point(30,180), FONT_HERSHEY_PLAIN, 2, Scalar(0,255,0), 2);

        Mat visual(curr_frame.rows, curr_frame.cols, CV_8UC3, Scalar(0,0,0));


        visualizer(visual, t_global.at<double>(0), t_global.at<double>(1));
        //cv::Mat verticalConcat;

        imshow("Visualizing", visual);

        imshow("Tracking", curr_frame);

        
        

        //imshow("Vertical Concat", verticalConcat);

        if (waitKey(1) == 27) break;

        prev_frame = curr_frame.clone();

    }

    cout << "Total processed frames: " << frame_count << endl;
    cout << "Total skipped frames due to insufficient matches: " << skip_frames << endl;
    cout << "Average time per frame: " << total_times / frame_count * 1000 << " ms" << endl;

    return 0;
}