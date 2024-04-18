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
