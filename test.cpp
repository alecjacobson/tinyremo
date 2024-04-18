#include "tinyremo.h"

// tictoc function to return time since last call in seconds
#include <chrono>
double tictoc()
{
  static std::chrono::time_point<std::chrono::high_resolution_clock> last = std::chrono::high_resolution_clock::now();
  auto now = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count() * 1e-6;
  last = now;
  return elapsed;
}

#include <Eigen/Core>
#include <Eigen/Cholesky>

// Benchmark ∂y/∂x for y = ‖ A(x)⁻¹ x ‖². This test chokes autodiff.github.io
// and https://github.com/Rookfighter/autodiff-cpp/
template <int N>
void call_and_time()
{
  using std::printf;
  using Scalar = double;
  tictoc();
  std::vector<Scalar> grads;
  Eigen::Matrix<Var<Scalar>, N, 1> x;
  const int max_iters = 1000;
  for(int iter = 0; iter < max_iters; ++iter)
  {
    Tape<Scalar> tape;
    // x has unregistered entries. Initialize them and put on tape
    for (int i = 0; i < x.rows(); ++i) 
    {
      x(i) = Var<Scalar>(&tape, tape.push_scalar(), i+1);
    }
    Eigen::Matrix<Var<Scalar>, N, N> A;
    for (int i = 0; i < A.rows(); ++i) 
    {
      for (int j = 0; j < A.cols(); ++j) 
      {
        A(i,j) = x(i) + x(j);
      }
      A(i,i) = A(i,i)+A(i,i);
    }
    Eigen::Matrix<Var<Scalar>, N, 1> b = A.llt().solve(x);
    Var<Scalar> y = b.array().square().sum();
    grads = y.grad();
  }
  double t_elapsed = tictoc();
  printf("%d %g\n",N,t_elapsed/max_iters);
  printf("  ... %% ");
  for (int i = 0; i < N; ++i) 
  {
    printf("%g ",grads[x(i).getIndex()]);
  }
  printf("\n");
}

// Generate a sequence of calls to call_and_time
template <int N, int M>
void call_and_time_all()
{
  if constexpr (N > M) 
  {
    return;
  }else
  {
    call_and_time<N>();
    call_and_time_all<N+1,M>();
  }
}


#include <iostream>

int main() {
  // Some simple tests
  {
    using Scalar = double;
    Tape<Scalar> tape;

    Var<Scalar> x(&tape, tape.push_scalar(), 0.5);
    Var<Scalar> y(&tape, tape.push_scalar(), 4.2);
    // This variable should not show up in sparse_grad
    Var<Scalar> a(&tape, tape.push_scalar(), 0.0/0.0);

    Var<Scalar> z = x * y + sin(x) + pow(x, y) + log(y) + exp(x) + cos(y);

    std::cout << "z = " << z.getValue() << std::endl;

    auto grads = z.grad();
    auto sparse_grads = z.sparse_grad();
    std::cout << "∂z/∂a " << (sparse_grads.find(a.getIndex()) == sparse_grads.end() ? "correctly not found" : "incorrectly found") << std::endl;
    std::cout << "∂z/∂x = " << sparse_grads[x.getIndex()] << std::endl;
    std::cout << "∂z/∂y = " << sparse_grads[y.getIndex()] << std::endl;
  }

  // Performance benchmark 
  call_and_time_all<1,16>();

  return 0;
}

