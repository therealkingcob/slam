#include <iostream>

using namespace std;

#include <ctime>

#include <Eigen/Core>

#include <Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50;

int main(int argc, char **argv) {

    Matrix<float, 2, 3> matrix_23;

    Vector3d v_3d;

    Matrix<float, 3,1> vd_3d;

    Matrix3d matrix_33 = Matrix3d::Zero();

    Matrix<double, Dynamic, Dynamic> matrix_dyanmic;

    MatrixXd matrix_x;

    matrix_23  << 1,2,3,4,5,6;

    cout << "matrix 23 from 1 to 6 \n" << matrix_23 << endl;

    cout << "print matrix 2x3: " << endl;

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) {
            cout << matrix_23(i, j) << "\t";
            cout << endl;
        }
    }


    cout << "wordls" << endl;
    return 0;
}