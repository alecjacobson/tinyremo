#include "tinyremo.h"
#include "tinyremo_eigen.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

template <typename Scalar>
Scalar spring_energy(const Eigen::Matrix<Scalar, Eigen::Dynamic, 2>& X, const Eigen::MatrixXi& E, const double l_rest)
{
  Scalar f = Scalar(0);
  for(int e = 0; e < E.rows(); ++e)
  {
    const auto& i = E(e,0);
    const auto& j = E(e,1);
    const auto& x_i = X.row(i);
    const auto& x_j = X.row(j);
    const auto& d_ij = x_i - x_j;
    const auto& l_ij = d_ij.norm();
    const auto& diff = l_ij - Scalar(l_rest);
    f += Scalar(0.5)*diff*diff;
  }
  return f;
}

template <int M, int N, bool print = false>
void mass_spring_sparse_hessian()
{
  Tape<double> tape_1;
  Tape<Var<double>> tape_2;
  // Spring mass positions
  Eigen::Matrix<Var<Var<double>>, Eigen::Dynamic, 2> X(M*N,2);
  for(int i = 0; i < M; ++i)
  {
    for(int j = 0; j < N; ++j)
    {
      X(i*N+j,0) = Var<Var<double>>(&tape_2, tape_2.push_scalar(), {&tape_1, tape_1.push_scalar(), i});
      X(i*N+j,1) = Var<Var<double>>(&tape_2, tape_2.push_scalar(), {&tape_1, tape_1.push_scalar(), j});
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

  Var<Var<double>> f = spring_energy(X, E, l_rest);

  // Dense first derivatives because we know that they're dense and that every
  // row of Hessian has at least one entry.
  auto df_d = f.grad();

  // We need to map X's elements' inner indices to indices in the rows/cols of
  // the Hessian. We'll use column major order here.
  std::map<size_t,size_t> outer_row,inner_col;
  for(int j = 0; j < X.cols(); ++j)
  {
    for(int i = 0; i < X.rows(); ++i)
    {
      inner_col[X(i,j).getValue().getIndex()] = X.rows()*j + i;
      outer_row[X(i,j).getIndex()] = X.rows()*j + i;
    }
  }

  // Collect sparse second derivatives
  std::vector<Eigen::Triplet<double>> triplets;
  for(int i = 0; i < X.rows(); ++i)
  {
    for(int j = 0;j < X.cols(); ++j)
    {
      auto d2f_dijd = df_d[X(i,j).getIndex()].sparse_grad();
      size_t outer_row_ij = outer_row[X(i,j).getIndex()];
      for(auto& [index, value] : d2f_dijd)
      {
        //size_t inner_col_ij = inner_col[key];
        if(inner_col.find(index) != inner_col.end())
        {
          size_t inner_col_ij = inner_col[index];
          triplets.emplace_back(outer_row_ij, inner_col_ij, value);
        }
      }
    }
  }

  Eigen::SparseMatrix<double> H(X.size(),X.size());
  H.setFromTriplets(triplets.begin(),triplets.end());

  if constexpr(print)
  {
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

    std::cout<<"H=sparse("<<H.rows()<<","<<H.cols()<<");"<<std::endl;
    for(int k = 0; k < H.outerSize(); ++k)
    {
      for(Eigen::SparseMatrix<double>::InnerIterator it(H,k); it; ++it)
      {
        std::cout<<"H("<<it.row()+1<<","<<it.col()+1<<")="<<it.value()<<";"<<" ";
      }
      std::cout<<std::endl;
    }
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
    const int M = 3;
    const int N = 2;
    std::cout<<"# 2D mass-spring system with "<<M<<"x"<<N<<" grid"<<std::endl;
    mass_spring_sparse_hessian<M,N,true>();
  }

  std::cout<<std::endl<<"# Benchmark"<<std::endl;
  call_and_time_all<2,128>();
}
