#include "tinyremo_eigen.h"
#include <tuple>
#include <type_traits>

using namespace tinyremo;

int main()
{


  {
    Tape<double> tape_1;
    Var<double> x = record_scalar<double>(0.5,tape_1);
  }
  {
    Tape<double> tape_1;
    Tape<Var<double>> tape_2;
    Var<Var<double>> x = record_scalar<double>(0.5,tape_1,tape_2);
    Var<Var<double>> y = record_scalar<double>(1.5,tape_1,tape_2);
    printf("(%g,%d,%d)\n",x.getValue().getValue(),x.getIndex(),x.getValue().getIndex());
    printf("(%g,%d,%d)\n",y.getValue().getValue(),y.getIndex(),y.getValue().getIndex());
  }

  Eigen::Matrix<double, 3,2> X;
  X << 0, 2,
       1, 4,
       9, 8;

  {
    Tape<double> tape_1;
    Eigen::Matrix<Var<double>, 3,2> Y = record_matrix(X, tape_1);
    for(int i = 0; i < Y.rows(); ++i)
    {
      for(int j = 0; j < Y.cols(); ++j)
      {
        printf("%g(%d) ",Y(i,j).getValue(),Y(i,j).getIndex());
      }
      printf("\n");
    }

    Eigen::Matrix<Var<double>, 3,2> X = Y.array().square();
    Var<double> z = Y.array().sum() + X.array().sum();

    auto [dzdY,dzdX] = gradient(z,Y,X);
    for(int i = 0; i < dzdY.rows(); ++i)
    {
      for(int j = 0; j < dzdY.cols(); ++j)
      {
        printf("%g ",dzdY(i,j).getValue());
      }
      printf("\n");
    }
  }
  {
    Tape<double> tape_1;
    Tape<Var<double>> tape_2;
    Eigen::Matrix<Var<Var<double>>, 3,2> Y = record_matrix(X, tape_1, tape_2);
    for(int i = 0; i < Y.rows(); ++i)
    {
      for(int j = 0; j < Y.cols(); ++j)
      {
        printf("%g(%d,%d) ",Y(i,j).getValue().getValue(),
            Y(i,j).getIndex(),
            Y(i,j).getValue().getIndex());
      }
      printf("\n");
    }
  }
}
