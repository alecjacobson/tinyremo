#include "tinyremo.h"
#include "tinyremo_eigen.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

using namespace tinyremo;
template <typename Scalar> 
Eigen::Matrix<Scalar,Eigen::Dynamic,1> spring_residuals(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 2>& X, const Eigen::MatrixXi& E, const double l_rest)
{
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> f(E.rows());
  for(int e = 0; e < E.rows(); ++e)
  {
    const auto& i = E(e,0);
    const auto& j = E(e,1);
    const auto& x_i = X.row(i);
    const auto& x_j = X.row(j);
    const auto& d_ij = x_i - x_j;
    const auto& l_ij = d_ij.norm();
    f(e) = l_ij - Scalar(l_rest);
  }
  return f;
}

template <typename Scalar>
Scalar spring_energy(const Eigen::Matrix<Scalar, Eigen::Dynamic, 2>& X, const Eigen::MatrixXi& E, const double l_rest)
{
  return Scalar(0.5) * spring_residuals(X,E,l_rest).squaredNorm();
}

template <int M, int N, bool print = false>
void mass_spring_sparse_hessian()
{
  // Spring mass positions
  Eigen::Matrix<double, Eigen::Dynamic, 2> X_double(M*N,2);
  for(int i = 0; i < M; ++i)
  {
    for(int j = 0; j < N; ++j)
    {
      X_double(i*N+j,0) = i;
      X_double(i*N+j,1) = j;
    }
  }
  // Spring rest length
  const double l_rest = 2.0;
  // Grid of spring edges
  Eigen::MatrixXi E(M*(N-1)+(M-1)*N,2);
  {
    int e = 0;
    for(int i = 0; i < M; ++i)
    {
      for(int j = 0; j < N; ++j)
      {
        if(j<N-1)
        {
          E.row(e++) << i*N+j, i*N+j+1;
        }
        if(i<M-1)
        {
          E.row(e++) << i*N+j, (i+1)*N+j;
        }
      }
    }
  }

  Tape<double> tape_1;
  Tape<Var<double>> tape_2;
  // Copy from double to Var<Var<double>> and record on tapes
  Eigen::Matrix<Var<Var<double>>, Eigen::Dynamic, 2> X = record_matrix(X_double, tape_1, tape_2);
  
  Var<Var<double>> f = spring_energy(X, E, l_rest);
  // Dense first derivatives because we know that they're dense and that every
  // row of Hessian has at least one entry.
  auto df_d = f.grad();
  auto H = sparse_hessian(f, X);

  if constexpr(print)
  {
    // If we're printing then let's also check the Jacobian
    // It's not required, but since Jacobian only needs first derivatives, let's
    // stip off the outer layer. This way J will contain doubles instead of
    // Var<double>.
    auto X_lower = X.unaryExpr([](const Var<Var<double>>& x){return x.getValue();}).eval();
    auto F = spring_residuals(X_lower,E,l_rest);
    auto J = sparse_jacobian(F,X_lower);

    // Print matlab friendly variables
    std::cout<<"X=["<<std::endl;
    for(int i = 0; i < X.rows(); ++i)
    {
      for(int j = 0;j < X.cols(); ++j)
      {
        std::cout<<X(i,j).getValue().getValue()<<" ";
      }
      std::cout<<";";
    }
    std::cout<<"];"<<std::endl;
    std::cout<<"E=["<<E<<"]+1;"<<std::endl;

    std::cout<<"df_dX=["<<std::endl;
    for(int i = 0; i < X.rows(); ++i)
    {
      for(int j = 0; j < X.cols(); ++j)
      {
        std::cout<<df_d[X(i,j).getIndex()].getValue()<<" ";
      }
      std::cout<<";";
    }
    std::cout<<"];"<<std::endl;

    const auto print_sparse_matrix = [](const Eigen::SparseMatrix<double>& A, const std::string name)
    {
      std::cout<<name<<" = sparse("<<A.rows()<<","<<A.cols()<<");"<<std::endl;
      for(int k = 0; k < A.outerSize(); ++k)
      {
        for(Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
        {
          std::cout<<name<<"("<<it.row()+1<<","<<it.col()+1<<")="<<it.value()<<";"<<" ";
        }
        std::cout<<std::endl;
      }
    };
    print_sparse_matrix(J,"J");
    print_sparse_matrix(H,"H");
    std::cout<<"% plot using gptoolbox"<<std::endl;
    std::cout<<"% tsurf(E,X);hold on;qvr(X,reshape((1e-10*speye(size(H)) + H)\\df_dX(:),size(X)));hold off;"<<std::endl;
  }
}


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

template <int N>
void call_and_time()
{
  using std::printf;
  tictoc();
  mass_spring_sparse_hessian<N,N>();
  printf("%d %g\n",N*N,tictoc());
}

template <int N, int M>
void call_and_time_all()
{
  if constexpr (N > M) 
  {
    return;
  }else
  {
    call_and_time<N>();
    call_and_time_all<2*N,M>();
  }
}

int main()
{
  // 2D mass-springs
  {
    const int M = 5;
    const int N = 4;
    std::cout<<"# 2D mass-spring system with "<<M<<"x"<<N<<" grid"<<std::endl;
    mass_spring_sparse_hessian<M,N,true>();
  }

  std::cout<<std::endl<<"# Benchmark"<<std::endl;
  call_and_time_all<2,128>();
}
