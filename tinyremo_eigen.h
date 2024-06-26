// Copyright (C) 2024 Alec Jacobson <alecjacobson@gmail.com>
#pragma once
#include "tinyremo.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <utility>

namespace Eigen
{
  template<typename T>
  struct NumTraits<tinyremo::Var<T>> : NumTraits<T> 
  {
    typedef tinyremo::Var<T> Real;
    typedef tinyremo::Var<T> Literal;
    enum {
      IsComplex = NumTraits<T>::IsComplex,
      IsInteger = 0,
      IsSigned = 1,
      RequireInitialization = 0,
      // Not sure if this should be an additive or multiplicative constant
      ReadCost = NumTraits<T>::ReadCost + 2,
      AddCost = NumTraits<T>::AddCost + 2,
      MulCost = NumTraits<T>::MulCost + 2
    };
  };
}

namespace tinyremo
{
  // ```
  // Eigen::Matrix<double, 3,2> X;
  // X << 0, 2,
  //      1, 4,
  //      9, 8;
  // Tape<double> tape_1;
  // Tape<Var<double>> tape_2;
  // Eigen::Matrix<Var<Var<double>>, 3,2> Y = record_matrix(X, tape_1, tape_2);
  // ```
  template< typename Derived, typename... Tapes>
  auto record_matrix(const Eigen::MatrixBase<Derived>& X, Tapes&... tapes) 
  {
    return X.unaryExpr([& ](auto&& x) { return record_scalar(x, tapes...); }).matrix().eval();
  }

  template<typename Matrix, typename Func>
  inline void iterateMatrix(const Matrix& X, const Func & func)
  {
    if constexpr (Matrix::IsRowMajor) {
      for (Eigen::Index i = 0; i < X.rows(); ++i)
        for (Eigen::Index j = 0; j < X.cols(); ++j)
          func(i, j);
    } else {
      for (Eigen::Index j = 0; j < X.cols(); ++j)
        for (Eigen::Index i = 0; i < X.rows(); ++i)
          func(i, j);
    }
  }

  // ```
  // auto [dfdX,dfdY] = gradient(z,X,Y);
  // ```
  template <typename Scalar, typename... Matrices>
  auto gradient(
    const Var<Scalar> & f,
    Matrices &... matrices)
  {
    auto grads = f.grad();
    auto compute_grad_matrix = [&grads](auto& matrix)
    {
      using MatrixType = std::decay_t<decltype(matrix)>;
      // Rebuild MatrixType with Var<Scalar> replaced by Scalar
      using GradMatrixType = Eigen::Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime, MatrixType::Options, MatrixType::MaxRowsAtCompileTime, MatrixType::MaxColsAtCompileTime>;
      GradMatrixType grad_matrix(matrix.rows(), matrix.cols());
      iterateMatrix(matrix, [&](Eigen::Index i, Eigen::Index j) { grad_matrix(i, j) = grads[matrix(i, j).getIndex()]; });
      return grad_matrix;
    };
    // Helper lambda to apply compute_grad_matrix to each matrix and return a
    // tuple of results
    auto tuple_of_gradients = [&](auto&... mats) {
      return std::make_tuple(compute_grad_matrix(mats)...);
    };
    return std::apply(tuple_of_gradients, std::tie(matrices...));
  }

  template < typename Matrix, typename Container>
  size_t collect_indices( const Matrix & X, Container & row, size_t index = 0)
  {
    // Depending on the containers this is non-trivial to parallelize
    iterateMatrix(X, [&](Eigen::Index i, Eigen::Index j)
      {
        auto k = X(i,j).getIndex();
        row[k] = index;
        ++index;
      });
    return index;
  }

  template < typename Matrix, typename OuterContainer, typename InnerContainer>
  size_t collect_outer_and_inner_indices(
      const Matrix & X,
      OuterContainer & outer_row,
      InnerContainer & inner_col,
      size_t index = 0)
  {
    // Depending on the containers this is non-trivial to parallelize
    iterateMatrix(X, [&](Eigen::Index i, Eigen::Index j)
      {
        auto k = X(i,j).getIndex();
        //assert(k < outer_row.size());
        if constexpr (std::is_same_v<OuterContainer, std::vector<size_t>>) { assert(k < outer_row.size()); }
        
        outer_row[k] = index;
        const auto & Xij = X(i,j);
        const auto & Xij_value = Xij.getValue();
        const auto & Xij_value_index = Xij_value.getIndex();
        //inner_col[X(i,j).getValue().getIndex()] = index;
        if constexpr (std::is_same_v<InnerContainer, index_map<size_t,size_t>>) { assert(inner_col.find(Xij_value_index) == inner_col.end()); }
        inner_col[Xij_value_index] = index;
        ++index;
      });
    return index;
  }

  template <typename... Matrices>
  size_t max_index( Matrices &... matrices)
  {
    size_t max_so_far = 0;
    auto max_index_helper = [&max_so_far](auto& X)
    {
      iterateMatrix(X, [&](Eigen::Index i, Eigen::Index j)
      {
        max_so_far = std::max(max_so_far, X(i,j).getIndex());
      });
    };
    (max_index_helper(matrices), ...);
    return max_so_far;
  }

  template <typename... Matrices>
  size_t max_value_index( Matrices &... matrices)
  {
    size_t max_so_far = 0;
    auto max_value_index_helper = [&max_so_far](auto& X)
    {
      iterateMatrix(X, [&](Eigen::Index i, Eigen::Index j)
      {
        max_so_far = std::max(max_so_far, X(i,j).getValue().getIndex());
      });
    };
    (max_value_index_helper(matrices), ...);
    return max_so_far;
  }

  template <typename Scalar, typename FType, typename InnerContainer>
  Eigen::SparseMatrix<Scalar> sparse_jacobian_from_indices(
    const FType & F,
    const InnerContainer & inner_col,
    const size_t max_index)
  {
    static_assert(std::is_same_v<typename FType::value_type, Var<Scalar>>);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    for(size_t i = 0; i < F.size(); ++i)
    {
      auto dFi_d = F[i].sparse_grad();
      for(auto& [index, value] : dFi_d)
      {
        //if(inner_col.find(index) != inner_col.end())
        //{
        //  size_t j = inner_col[index];
        // inner_col may be const
        auto it = inner_col.find(index);
        if(it != inner_col.end())
        {
          size_t j = it->second;
          triplets.emplace_back(i, j, value);
        }
      }
    };
    Eigen::SparseMatrix<Scalar> J(F.size(), max_index);
    J.setFromTriplets(triplets.begin(),triplets.end());
    return J;
  }

  template <typename Scalar, typename FType, typename... Matrices>
  Eigen::SparseMatrix<Scalar> sparse_jacobian_generic(
    const FType & F,
    Matrices &... matrices)
  {
    index_map<size_t,size_t> col;
    int index = 0;
    auto collect_all_indices = [&index,&col](auto& X) 
    {
      index = collect_indices(X, col , index);
    };
    (collect_all_indices(matrices), ...);
    return sparse_jacobian_from_indices< Scalar>(F, col, index);
  }

  template <typename Scalar, typename... Matrices>
  Eigen::SparseMatrix<Scalar> sparse_jacobian(
    const std::vector<Var<Scalar>> & F,
    Matrices &... matrices)
  {
    return sparse_jacobian_generic<Scalar>(F, matrices...);
  }

  template <
    typename Scalar,
    int Rows,
    int RowsAtCompileTime,
    typename... Matrices>
  Eigen::SparseMatrix<Scalar> sparse_jacobian(
    const Eigen::Matrix<Var<Scalar>, Rows, 1, Eigen::ColMajor, RowsAtCompileTime, 1 > & F,
    Matrices &... matrices)
  {
    return sparse_jacobian_generic<Scalar>(F, matrices...);
  }

  template<typename... Matrices>
  constexpr int any_dynamic() {
      return ((Matrices::ColsAtCompileTime == Eigen::Dynamic || Matrices::RowsAtCompileTime == Eigen::Dynamic) || ...);
  }

  // retrun Eigen::RowMajor if all Matrices are RowMajor, otherwise Eigen::ColMajor
  template<typename... Matrices>
  constexpr int row_major_if_all() {
    return ((Matrices::IsRowMajor) && ...) ? Eigen::RowMajor : Eigen::ColMajor;
  }

  // Assuming non are dynamic
  template<typename... Matrices>
  constexpr int total_compile_time_size() {
    // Better to just assume otherwise logic get's annoying later
    //static_assert(!any_dynamic<Matrices...>());
    return (0 + ... + (Matrices::ColsAtCompileTime * Matrices::RowsAtCompileTime));
  }

  template <typename Scalar, typename... Matrices>
  auto hessian(const Var<Var<Scalar>> & f, Matrices &... matrices)
  {
    constexpr bool has_dynamic = any_dynamic<Matrices...>();
    constexpr int compile_time_size = total_compile_time_size<Matrices...>();
    constexpr int N = has_dynamic ? Eigen::Dynamic : compile_time_size;
    constexpr int order = row_major_if_all<Matrices...>();
    int N_run = N;
    if constexpr (N == Eigen::Dynamic) 
    {
      N_run = (0 + ... + (matrices.cols() + matrices.rows()));
    }
    Eigen::Matrix<Scalar,N,N,order> H(N_run, N_run);

    // Dense gradient
    auto df_d_raw = f.grad();
    int outer_i = 0;
    auto outer_loop = [&](auto & X)
    {
      iterateMatrix(X, [&](Eigen::Index i, Eigen::Index j)
      {
        auto df_dij = df_d_raw[X(i,j).getIndex()];
        auto d2f_dij_d_raw = df_dij.grad();

        int inner_j = 0;
        auto inner_loop = [&](auto & Y)
        {
          iterateMatrix(Y, [&](Eigen::Index k, Eigen::Index l)
          {
            auto d2f_dij_dkl = d2f_dij_d_raw[Y(k,l).getValue().getIndex()];
            H(outer_i, inner_j) = d2f_dij_dkl;
            inner_j++;
          });
        };
        (inner_loop(matrices), ...);

        outer_i++;
      });
    };
    (outer_loop(matrices), ...);

    return H;
  }

  // Assume that the Hessian is sparse but every row or column has at least one
  // entry.
  template <typename Scalar, typename... Matrices>
  Eigen::SparseMatrix<Scalar> sparse_hessian(
    const Var<Var<Scalar>> & f,
    Matrices &... matrices)
  {
    // Dense gradient
    auto df_d_raw = f.grad();
    std::vector<size_t> outer_row(max_index(matrices...)+1);
    index_map<size_t,size_t> inner_col;
    int index = 0;
    auto collect_indices = [&inner_col,&outer_row,&index](auto& X) 
    {
      index = collect_outer_and_inner_indices(X, outer_row, inner_col, index);
    };
    (collect_indices(matrices), ...);

    // Filter df_d to only include the indices that are in the outer_row
    std::vector<Var<Scalar>> df_d(index);
    auto collect_df_d_entries = [&](auto&X)
    {
      // This implements
      //   for each i; A[J[i]] = B[i]
      // Would it be better to have
      //   for each j: A[j] = B[I[j]]
      iterateMatrix(X, [&](Eigen::Index i, Eigen::Index j)
      {
        auto df_dij = df_d_raw[X(i,j).getIndex()];
        size_t outer_row_ij = outer_row[X(i,j).getIndex()];
        df_d[outer_row_ij] = df_dij;
      });
    };
    (collect_df_d_entries(matrices), ...);
    return sparse_jacobian_from_indices<Scalar>(df_d, inner_col, index);
  }
}
