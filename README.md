# 🥁 TinyReMo 🥁

Tiny header-only, minimal dependency reverse-mode automatic differentiation library for C++.

Based on the tape-based implmenetation tutorial at https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

## Compile and Run Tests

```bash 
clang++ -std=c++20 -I . -I [path/to/eigen] -I -o test test.cpp
./test
```

## Gradient of a function of scalar variables

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
