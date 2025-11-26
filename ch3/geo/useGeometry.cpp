#include <iostream>
#include <ctime>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc, char **argv) {

    Matrix3d rotation_matrix = Matrix3d::Identity();


    //rotating pi/4 or 45 degrees around the z-axis
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0,0,1));

    cout.precision(3);

    cout << "rotation matrix = \n " << rotation_vector.matrix() << endl; 

    rotation_matrix = rotation_vector.toRotationMatrix();

    Vector3d v(1, 0, 0);

    Vector3d v_rotated = rotation_vector * v;

    cout << "(1, 0, 0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;

    v_rotated = rotation_matrix * v;

    cout << "(1, 0 , 0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

    Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);

    cout << "yaw, pitch, roll = " << euler_angles.transpose() << endl; 



    cout << "test" << endl;
}