#include "tinyremo.h"

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>

int main() {
  // Some simple tests with first derivatives
  {
    printf("First derivatives\n");
    using Scalar = double;
    Tape<Scalar> tape;

    Var<Scalar> x(&tape, tape.push_scalar(), 0.5);
    Var<Scalar> y(&tape, tape.push_scalar(), 4.2);
    // This variable should not show up in sparse_grad
    Var<Scalar> a(&tape, tape.push_scalar(), 0.0/0.0);

    Var<Scalar> z = x * y + sin(x) + pow(x, y) + log(y) + exp(x) + cos(y);
    //Var<Scalar> z = x / y;

    std::cout << "z = " << z.getValue() << std::endl;

    auto grads = z.grad();
    auto sparse_grads = z.sparse_grad();
    std::cout << "∂z/∂a " << (sparse_grads.find(a.getIndex()) == sparse_grads.end() ? "correctly not found" : "incorrectly found") << std::endl;
    std::cout << "∂z/∂x = " << sparse_grads[x.getIndex()] << std::endl;
    std::cout << "∂z/∂y = " << sparse_grads[y.getIndex()] << std::endl;
  }


  // Second derivatives
  {
    printf("Second derivatives\n");
    Tape< double > tape_1;
    Tape< Var<double> > tape_2;
    Var<Var<double>> x(&tape_2, tape_2.push_scalar(), {&tape_1, tape_1.push_scalar(), 0.5});
    Var<Var<double>> y(&tape_2, tape_2.push_scalar(), {&tape_1, tape_1.push_scalar(), 4.2});
    //Var<Var<double>> z = x/y;
    auto z = x * y + sin(x) + pow(x, y) + log(y) + exp(x) + cos(y);
    auto dzd = z.grad();
    printf("z = %g\n", z.getValue().getValue());
    Var<double> dzdx = dzd[x.getIndex()];
    printf("∂z/∂x = %g\n", dzdx.getValue());
    Var<double> dzdy = dzd[y.getIndex()];
    printf("∂z/∂y = %g\n", dzdy.getValue());
    auto d2zdxd = dzdx.grad();
    printf("∂²z/∂x² = %g\n", d2zdxd[x.getValue().getIndex()]);
    printf("∂²z/∂x∂y = %g\n", d2zdxd[y.getValue().getIndex()]);
    auto d2zdyd = dzdy.grad();
    printf("∂²z/∂y∂x = %g\n", d2zdyd[x.getValue().getIndex()]);
    printf("∂²z/∂y² = %g\n", d2zdyd[y.getValue().getIndex()]);
  }

  // Second derivatives with Eigen
  {
    printf("Second derivatives with Eigen\n");
    Tape< double > tape_1;
    Tape< Var<double> > tape_2;
    const int N = 2;
    Eigen::Matrix<Var<Var<double>>, N, 1> x;
    for (int i = 0; i < x.rows(); ++i) 
    {
      x(i) = Var<Var<double>>(&tape_2, tape_2.push_scalar(), {&tape_1, tape_1.push_scalar(), i+1});
    }
    Eigen::Matrix<Var<Var<double>>, N, N> A;
    for (int i = 0; i < A.rows(); ++i) 
    {
      for (int j = 0; j < A.cols(); ++j) 
      {
        A(i,j) = x(i) + x(j);
      }
      A(i,i) = A(i,i)+A(i,i);
    }
    Eigen::Matrix<Var<Var<double>>, N, 1> b = A.llt().solve(x);
    Var<Var<double>> y = b.array().square().sum();
    auto dyd = y.grad();
    for (int i = 0; i < N; ++i) 
    {
      printf("∂y/∂x(%d) = %g\n", i, dyd[x(i).getIndex()].getValue());
    }
    for (int i = 0; i < N; ++i) 
    {
      auto d2ydxi = dyd[x(i).getIndex()].grad();
      for (int j = 0; j < N; ++j) 
      {
        printf("∂²y/∂x(%d)∂x(%d) = %g\n", i, j, d2ydxi[x(j).getValue().getIndex()]);
      }
    }
  }

}
