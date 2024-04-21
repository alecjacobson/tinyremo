#include "tinyremo.h"
#include "tinyremo_eigen.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>

using namespace tinyremo;
// tictoc function to return time since last call in seconds
double tictoc()
{
  static std::chrono::time_point<std::chrono::high_resolution_clock> last = std::chrono::high_resolution_clock::now();
  auto now = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count() * 1e-6;
  last = now;
  return elapsed;
}


// Benchmark ∂y/∂x for y = ‖ A(x)⁻¹ x ‖². This test chokes autodiff.github.io
// and https://github.com/Rookfighter/autodiff-cpp/
template <int N>
void call_and_time_llt()
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
void call_and_time_all_llt()
{
  if constexpr (N > M) 
  {
    return;
  }else
  {
    call_and_time_llt<N>();
    call_and_time_all_llt<N+1,M>();
  }
}


template <typename Scalar> 
Eigen::Matrix<Scalar,Eigen::Dynamic,1> spring_residuals(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 2>& X, const Eigen::MatrixXi& E, const double l_rest)
{
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> f(E.rows());
  for(int e = 0; e < E.rows(); ++e)
  {
    f(e) = (X.row(E(e,0)) - X.row(E(e,1))).norm() - Scalar(l_rest);
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
        if(j<N-1) { E.row(e++) << i*N+j, i*N+j+1; }
        if(i<M-1) { E.row(e++) << i*N+j, (i+1)*N+j; }
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


template <int N>
void call_and_time_mass_spring()
{
  using std::printf;
  tictoc();
  mass_spring_sparse_hessian<N,N>();
  printf("%d %g\n",N*N,tictoc());
}

template <int N, int M>
void call_and_time_all_mass_spring()
{
  if constexpr (N > M) 
  {
    return;
  }else
  {
    call_and_time_mass_spring<N>();
    call_and_time_all_mass_spring<2*N,M>();
  }
}

int main() {
  // Some simple tests
  {
    printf("# Some simple tests\n");
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
  {
    printf("\n# Performance Benchmark with Eigen::Cholesky\n");
    call_and_time_all_llt<1,16>();
  }

  // Second derivatives
  {
    printf("\n# Second derivatives\n");
    Tape< double > tape_1;
    Tape< Var<double> > tape_2;
    Var<Var<double>> x(&tape_2, tape_2.push_scalar(), {&tape_1, tape_1.push_scalar(), 0.5});
    Var<Var<double>> y(&tape_2, tape_2.push_scalar(), {&tape_1, tape_1.push_scalar(), 4.2});
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
    printf("\n# Second derivatives with Eigen\n");
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

  {
    const int M = 3;
    const int N = 2;
    std::cout<<"# 2D mass-spring system with "<<M<<"x"<<N<<" grid"<<std::endl;
    mass_spring_sparse_hessian<M,N,true>();
  }
  {
    std::cout<<std::endl<<"# Benchmark"<<std::endl;
    call_and_time_all_mass_spring<2,128>();
  }

  return 0;
}

