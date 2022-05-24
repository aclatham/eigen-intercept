#include <iostream>
#include <Eigen/Dense>
//#include "eigen-intercept.h"
 
using namespace Eigen;
int main()
{
   MatrixXf mat = MatrixXf::Random(256,256);
   MatrixXf mat1 = MatrixXf::Random(256,256);
   //std::cout << "Here is the adjoint of m:" << std::endl << mat << std::endl;
   //std::cout << mat << std::endl << std::endl;
   //std::cout << mat1 << std::endl;

   mat= mat*mat1;
}
