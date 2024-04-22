#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include "tinyremo_eigen.h"

#include <TinyAD/Scalar.hh>


#include <Eigen/Dense>
// Only really supports first derivatives: issues with abs() for nested
// AutoDiffScalar
#include <unsupported/Eigen/AutoDiff>
#include <iostream>
#include <chrono>
#include <vector>

const int grad_max_N = 64;
const int hess_max_N = 64;
const int max_vector_N = 21;
const int max_matrix_N = max_vector_N;
// macro to use N if N < max_N otherwise Eigen::Dynamic
#define CLAMP(N,M) (N < M ? N : Eigen::Dynamic)
#define VCLAMP(N) CLAMP(N,max_vector_N)
#define MCLAMP(N) CLAMP(N,max_matrix_N)

// Define a utility function to measure execution time
double tictoc()
{
    double t = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    static double t0 = t;
    double tdiff = t - t0;
    t0 = t;
    return tdiff;
}

// The original function modified to work with Eigen's AutoDiff
template <typename T, int N, int xN = VCLAMP(N)>
T llt_func(Eigen::Matrix<T, xN , 1>& x)
{
    Eigen::Matrix<T, MCLAMP(N), MCLAMP(N) > A(N,N);
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A(i, j) = x(i) + x(j);
        }
        A(i, i) = T(2.0) * A(i, i);
    }
    Eigen::Matrix<T, VCLAMP(N), 1> b = A.llt().solve(x);
    T y = b.squaredNorm();
    return y;
}


// Function to compute the gradient and Hessian of llt_func
template <int N>
void eigen_computeGradient(
    const Eigen::Matrix<double, VCLAMP(N), 1> & x,
          Eigen::Matrix<double, VCLAMP(N), 1> & grad)
{
  using AutoDiffScalar = Eigen::AutoDiffScalar<Eigen::VectorXd>;
  Eigen::Matrix<AutoDiffScalar, VCLAMP(N), 1> x_autodiff(N);
  for (int i = 0; i < N; ++i)
  {
    x_autodiff(i).value() = x(i);
    x_autodiff(i).derivatives() = Eigen::VectorXd::Unit(N, i);
  }

  // Compute gradient
  auto y_autodiff = llt_func<AutoDiffScalar, N>(x_autodiff);
  grad = y_autodiff.derivatives();
}

// Function to compute the gradient and Hessian of llt_func
template <int N, bool compute_hessian=true>
void tinyad_computeGradientAndHessian(
    const Eigen::Matrix<double, VCLAMP(N), 1> & _x,
    Eigen::Matrix<double, VCLAMP(N), 1> & grad,
    Eigen::Matrix<double, MCLAMP(N), MCLAMP(N)> & hessian)
{
  // Choose autodiff scalar type for 3 variables
  using ADouble = TinyAD::Double<N>;
  //// Example usage:
  //// Init a 3D vector of active variables and a 3D vector of passive variables
  //Eigen::Vector3<ADouble> x = ADouble::make_active({0.0, -1.0, 1.0});
  //Eigen::Vector3<double> y(2.0, 3.0, 5.0);
  //// Compute angle using Eigen functions and retrieve gradient and Hessian w.r.t. x
  //ADouble angle = acos(x.dot(y) / (x.norm() * y.norm()));
  //Eigen::Vector3d g = angle.grad;
  //Eigen::Matrix3d H = angle.Hess;
  Eigen::Matrix<ADouble, N, 1> x = ADouble::make_active(_x.template head<N>().eval());
  auto y = llt_func<ADouble, N, N>(x);
  grad = y.grad;
  if constexpr (compute_hessian)
  {
    hessian = y.Hess;
  }
}

// Function to compute the gradient and Hessian of llt_func
template <int N, bool compute_hessian=true>
void tinyremo_computeGradientAndHessian(
    const Eigen::Matrix<double, VCLAMP(N), 1> & _x,
    Eigen::Matrix<double, VCLAMP(N), 1> & grad,
    Eigen::Matrix<double, MCLAMP(N), MCLAMP(N)> & hessian)
{
  tinyremo::Tape<double> tape_1;
  tinyremo::Tape<tinyremo::Var<double>> tape_2;
  auto x = tinyremo::record_matrix(_x, tape_1, tape_2);
  auto y = llt_func<tinyremo::Var<tinyremo::Var<double>>, N>(x);
  auto [dydx] = tinyremo::gradient(y, x);
  grad = dydx.unaryExpr([&](auto&& x) { return x.getValue(); });
  if constexpr (compute_hessian)
  {
    hessian = tinyremo::hessian(y, x);
  }
}

// Use centered finite differences with epsilon=1e-5
template <int N>
void fd_computeGradientAndHessian(
    const Eigen::Matrix<double, VCLAMP(N), 1> & x,
          Eigen::Matrix<double, VCLAMP(N), 1> & grad,
          Eigen::Matrix<double, MCLAMP(N), MCLAMP(N) > & hessian)
{
  const double epsilon = 1e-5;
  auto computeGradient = [&](const Eigen::Matrix<double, VCLAMP(N), 1>& x) -> Eigen::Matrix<double, VCLAMP(N), 1>
  {
    Eigen::Matrix<double, VCLAMP(N), 1> grad(N);
    for (int i = 0; i < N; ++i)
    {
      Eigen::Matrix<double, VCLAMP(N), 1> x_plus = x;
      x_plus(i) += epsilon;
      Eigen::Matrix<double, VCLAMP(N), 1> x_minus = x;
      x_minus(i) -= epsilon;
      grad(i) = (llt_func<double,N>(x_plus) - llt_func<double,N>(x_minus)) / (2.0 * epsilon);
    }
    return grad;
  };
  grad = computeGradient(x);

  // Compute hessian as finite differences of the gradient
  hessian.setZero();
  for (int i = 0; i < N; ++i)
  {
    Eigen::Matrix<double, VCLAMP(N), 1> x_plus = x;
    x_plus(i) += epsilon;
    Eigen::Matrix<double, VCLAMP(N), 1> x_minus = x;
    x_minus(i) -= epsilon;
    hessian.col(i) = (computeGradient(x_plus) - computeGradient(x_minus)) / (2.0 * epsilon);
  }
}

// Helper function to append a value to a vector and return the new vector
template <typename T>
std::vector<T> appendToVector(const std::vector<T>& vec, T value) {
    std::vector<T> newVec(vec.begin(), vec.end());
    newVec.push_back(value);
    return newVec;
}

// Benchmark function adapted for C++20 templated lambdas, now returning a vector of times
template <int N, int max_N, auto Func>
auto benchmark() -> std::vector<double>
{
    tictoc(); // Start timer
    const int max_iter = 100;//1000/N;
    for(int iter = 0; iter < max_iter; iter++)
    {
        Func.template operator()<N>(); // Invoke the templated lambda
    }

    double time = tictoc()/max_iter; // Measure time for this N

    if constexpr (N < max_N)
    {
        auto times = benchmark<N*2, max_N, Func>(); // Recursive call to benchmark the next value of N
        return appendToVector(times, time); // Append current time and return
    }
    else
    {
        return {time}; // Return vector with single element for the base case
    }
}

int main()
{
  {
    const int N = 3;
    printf("# Outputs for N=%d\n",N);
    Eigen::Matrix<double, VCLAMP(N), 1> x(N), grad(N);
    Eigen::Matrix<double, MCLAMP(N), MCLAMP(N)> hessian(N,N);
    for (int i = 0; i < N; ++i) x(i) = i + 1.0;

    grad.setZero(); hessian.setZero();
    fd_computeGradientAndHessian<N>(x, grad, hessian);
    std::cout<<"       fd gradient: "<<grad.transpose()<<std::endl;
    std::cout<<"        fd hessian: "<<hessian.reshaped(1, N*N)<<std::endl;

    grad.setZero(); hessian.setZero();
    eigen_computeGradient<N>(x, grad);
    std::cout<<"   Eigen gradient: "<<grad.transpose()<<std::endl;

    grad.setZero(); hessian.setZero();
    tinyad_computeGradientAndHessian<N,true>(x, grad, hessian);
    std::cout<<"  tinyad gradient: "<<grad.transpose()<<std::endl;
    std::cout<<"   tinyad hessian: "<<hessian.reshaped(1, N*N)<<std::endl;

    grad.setZero(); hessian.setZero();
    tinyremo_computeGradientAndHessian<N,true>(x, grad, hessian);
    std::cout<<"tinyremo gradient: "<<grad.transpose()<<std::endl;
    std::cout<<" tinyremo hessian: "<<hessian.reshaped(1, N*N)<<std::endl;
  }


  // Define as a function that takes template parameter compute_hessian
  auto benchmark_func = []<bool compute_hessian, int min_N = 2, int max_N = compute_hessian ? hess_max_N : grad_max_N>()
  {
    auto eigen_func = []<int N>() {
      Eigen::Matrix<double, VCLAMP(N), 1> x(N), grad(N);
      Eigen::Matrix<double, MCLAMP(N), MCLAMP(N)> hessian(N,N);
      for (int i = 0; i < N; ++i) x(i) = i + 1.0;
      eigen_computeGradient<N>(x, grad);
    };

    auto tinyad_func = []<int N>() {
      Eigen::Matrix<double, VCLAMP(N), 1> x(N), grad(N);
      Eigen::Matrix<double, MCLAMP(N), MCLAMP(N)> hessian(N,N);
      for (int i = 0; i < N; ++i) x(i) = i + 1.0;
      tinyad_computeGradientAndHessian<N,compute_hessian>(x, grad, hessian);
    };

    auto tinyremo_func = []<int N>() {
      Eigen::Matrix<double, VCLAMP(N), 1> x(N), grad(N);
      Eigen::Matrix<double, MCLAMP(N), MCLAMP(N)> hessian(N,N);
      for (int i = 0; i < N; ++i) x(i) = i + 1.0;
      tinyremo_computeGradientAndHessian<N,compute_hessian>(x, grad, hessian);
    };

    //auto eigen_times = benchmark<min_N, max_N, eigen_func>();
    // only call eigen_func if compute_hessian=false;
    std::vector<double> eigen_times; if constexpr (!compute_hessian) { eigen_times = benchmark<min_N, max_N, eigen_func>(); }
    auto tinyad_times = benchmark<min_N, max_N, tinyad_func>();
    auto tinyremo_times = benchmark<min_N, max_N, tinyremo_func>();


    // TinyAD has to make active variables on the stack so this needs to be
    // pretty small.
    printf("# Gradient");
    if constexpr (compute_hessian) { printf(" and Hessian"); }
    printf(" Timings for N=%d…%d\n", 2, max_N);

    for (size_t i = 0; i < tinyremo_times.size(); ++i)
    {
      size_t ri = tinyremo_times.size()-i-1;
      if constexpr (compute_hessian) {
        printf("%lu %g %g\n", 
            min_N<<i,
            tinyad_times[ri],
            tinyremo_times[ri]);
      } else {
        printf("%lu %g %g %g\n", 
            min_N<<i,
            eigen_times[ri],
            tinyad_times[ri],
            tinyremo_times[ri]);
      }
    }
  };
  benchmark_func.template operator()<false>();
  benchmark_func.template operator()<true>();


  //return 0;
}

