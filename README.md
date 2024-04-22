# 🥁 TinyReMo 🥁

Tiny header-only, minimal dependency reverse-mode automatic differentiation library for C++.

Based on the tape-based rust implementation in the tutorial at https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

## Compile and Run Tests

```bash 
clang++ -std=c++20 -I . -I [path/to/eigen] -I -o test test.cpp
./test
```

## Gradient of a function of scalar variables

Variables for which derivatives are required must be registered with a common `Tape`.

```cpp
#include "tinyremo.h"
int main()
{
  using namespace tinyremo;

  Tape<double> tape;

  Var<double> x(&tape, tape.push_scalar(), 0.5);
  Var<double> y(&tape, tape.push_scalar(), 4.2);

  Var<double> z = x * y + sin(x) + pow(x, y) + log(y) + exp(x) + cos(y);

  std::cout << "z = " << z.getValue() << std::endl;

  auto grads = z.grad();
  std::cout << "∂z/∂x = " << grads[x.getIndex()] << std::endl;
  std::cout << "∂z/∂y = " << grads[y.getIndex()] << std::endl;
}
```


## As a custom scalar type for Eigen matrices

```cpp
Tape<Scalar> tape;
Eigen::Matrix<Var<Scalar>, N, 1> x;
// x has unregistered entries. Initialize them and put on tape
for (int i = 0; i < x.rows(); ++i) 
{
  x(i) = Var<Scalar>(&tape, tape.push_scalar(), i+1);
}
```

## Higher-order derivatives

For each derivative we wish to take we must create a new `Tape`.

```cpp
Tape< double > inner_tape;
Tape< Var<double> > outer_tape;

// Inner variable
Var<double> x_inner(&inner_tape, inner_tape.push_scalar(), 0.5);
// Outer variable (we'll actually use this one)
Var<Var<double>> x(&outer_tape, outer_tape.push_scalar(), x_inner);
// Construct outer variable in one line
Var<Var<double>> y(&outer_tape, outer_tape.push_scalar(), {&inner_tape, inner_tape.push_scalar(), 4.2});

auto z = x * y + sin(x) + pow(x, y) + log(y) + exp(x) + cos(y);

// First derivatives
auto dzd = z.grad();
printf("z = %g\n", z.getValue().getValue());
Var<double> dzdx = dzd[x.getIndex()];
printf("∂z/∂x = %g\n", dzdx.getValue());
Var<double> dzdy = dzd[y.getIndex()];
printf("∂z/∂y = %g\n", dzdy.getValue());

// Second derivatives
auto d2zdxd = dzdx.grad();
printf("∂²z/∂x² = %g\n", d2zdxd[x.getValue().getIndex()]);
printf("∂²z/∂x∂y = %g\n", d2zdxd[y.getValue().getIndex()]);
auto d2zdyd = dzdy.grad();
printf("∂²z/∂y∂x = %g\n", d2zdyd[x.getValue().getIndex()]);
printf("∂²z/∂y² = %g\n", d2zdyd[y.getValue().getIndex()]);
```

## Easier construction of recorded variables

Call `record_scalar` to record a value on a bunch of tapes (in order) to create
appropriately nested `Var` object in one line. In this case, `x` will be a
twice-differentiable `Var<Var<double>>`.

```cpp
Tape<double> tape_1;
Tape<Var<double>> tape_2;
auto x = record_scalar(0.5,tape_1,tape_2);
```

Similarly, for Eigen matrices, `record_matrix` will copy `double` values from a
matrix and record them on a bunch of tapes in order to create a matrix of `Var`
objects. In this case, `Y` will be a twice-differentiable
`Eigen::Matrix<Var<Var<double>>,Eigen::Dynamic,Eigen::Dynamic>`.

```cpp
Eigen::MatrixXd X = Eigen::MatrixXd::Random(3, 3);
Tape<double> tape_1;
Tape<Var<double>> tape_2;
auto Y = record_matrix(X,tape_1,tape_2);
```

## Gradients with respect to Eigen matrices as Eigen matrices

For a once-differentiable `Var<double>` object `f`, `f.grad()` will return a
`std::vector<double>` of the gradient of `f` with respect to _everything_ on the
tape, including auxiliary variables. If `f` is a function of some
`Eigen::Matrix` types then we can call `gradient` to get the gradient of `f` in
appropriately sized matrices.

```cpp
Tape<double> tape;
Eigen::MatrixXd X1 = Eigen::MatrixXd::Random(3, 3);
Eigen::MatrixXd X2 = Eigen::MatrixXd::Random(3, 3);
auto Y1 = record_matrix(X1,tape);
auto Y2 = record_matrix(X2,tape);
auto f = Y1.array().sin().sum() * Y2.array().cos().sum();
auto [dfdY1, dfdY2] = gradient(f, Y1, Y2);
```

## Hessian with respect to Eigen matrices 
For a twice-differentiable scalar `Var<double>` object `f`, `hessian` will
populate a `Eigen::Matrix<double,…>` with the Hessian of `f`. The rows and columns will correspond
to second partial derivatives with respect to pairs of elements of the
matrices provided as input in their respective storage orders (`Eigen::RowMajor`
or `Eigen::ColMajor`). The output matrix will be `Eigen::ColMajor` unless all inputs
are `Eigen::RowMajor`. The output matrix will have `Eigen::Dynamic` rows and
cols, unless all inputs have fixed sizes.

```cpp
Tape<double> tape;
Tape<Var<double>> tape_2;
Eigen::MatrixXd X1 = Eigen::MatrixXd::Random(3, 3);
Eigen::MatrixXd X2 = Eigen::MatrixXd::Random(3, 3);
auto Y1 = record_matrix(X1,tape_1,tape_2);
auto Y2 = record_matrix(X2,tape_1,tape_2);
auto f = Y1.array().sin().sum() * Y2.array().cos().sum();
auto H = gradient(f, Y1, Y2);
```

## Sparse Jacobian as `Eigen::SparseMatrix`

For a once-differentiable vector of `Var<double>` objects `F`, `sparse_jacobian`
will populate a `Eigen::SparseMatrix<double>` with the Jacobian of `F`. The rows
correspond to the elements of `F` and the columns correspond to the elements of
the matrices provided as input in their respective storage orders
(`Eigen::RowMajor` or `Eigen::ColMajor`).

```cpp
Tape<double> tape;
Eigen::MatrixXd X1 = Eigen::MatrixXd::Random(3, 3);
Eigen::MatrixXd X2 = Eigen::MatrixXd::Random(3, 3);
auto Y1 = record_matrix(X1,tape);
auto Y2 = record_matrix(X2,tape);
// expression involving colwise sum resulting in a column matrix
auto F = (Y1.array().sin().colwise().sum() * Y2.array().cos().colwise().sum()).matrix();
auto J = sparse_jacobian(F, Y1, Y2);
```

## Sparse Hessian as `Eigen::SparseMatrix`

Similarly, for a twice-differentiable `Var<Var<double>>` object `f`,
`sparse_hessian` will populate a `Eigen::SparseMatrix<double>` with the Hessian
of `f`. The rows and columns correspond to the elements of `f` to the elements
of input matrices in their respective storage orders (`Eigen::RowMajor` or
`Eigen::ColMajor`).

```cpp
Tape<double> tape_1;
Tape<Var<double>> tape_2;
auto Y1 = record_matrix(X1,tape);
auto Y2 = record_matrix(X2,tape);
auto f = Y1.array().sin().sum() * Y2.array().cos().sum();
auto H = sparse_hessian(f, Y1, Y2);
```
