#include "tinyremo.h"

#include <Eigen/Core>

#include <utility>

namespace Eigen
{
  template<typename T>
  struct NumTraits<Var<T>> : NumTraits<T> 
  {
    typedef Var<T> Real;
    typedef Var<T> Literal;
  
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

// ```
// Tape<double> tape_1;
// Tape<Var<double>> tape_2;
// Var<Var<double>> x = make_scalar<double>(0.5,tape_1,tape_2);
// ```
template<typename Scalar, typename FirstTape, typename... RestTapes>
auto make_scalar(Scalar value, FirstTape& first_tape, RestTapes&... rest_tapes)
{
  Var<Scalar> x(&first_tape, first_tape.push_scalar(), value);
  if constexpr (sizeof...(RestTapes) == 0)
  { 
    return x; 
  }else
  {
    return make_scalar(x, rest_tapes...);
  }
}

// ```
// Eigen::Matrix<double, 3,2> X;
// X << 0, 2,
//      1, 4,
//      9, 8;
// Tape<double> tape_1;
// Tape<Var<double>> tape_2;
// Eigen::Matrix<Var<Var<double>>, 3,2> Y = make_matrix(X, tape_1, tape_2);
// ```
template< typename Derived, typename... Tapes>
auto make_matrix(const Eigen::MatrixBase<Derived>& X, Tapes&... tapes) 
{
  return X.unaryExpr([& ](auto&& x) { return make_scalar(x, tapes...); });
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
  // grads[X(i,j).getIndex()] will give the gradient of f with respect to the
  // scalar element X(i,j) of the matrix X

  // return type will be a tuple of grad matrices the same sizes as each input
  // in matrices

  // Helper lambda to compute the gradient matrix for a single input matrix
    auto compute_grad_matrix = [&grads](auto& matrix) {
        using MatrixType = std::decay_t<decltype(matrix)>;
        MatrixType grad_matrix(matrix.rows(), matrix.cols());
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                grad_matrix(i, j) = grads[matrix(i, j).getIndex()];
            }
        }
        return grad_matrix;
    };

    // Helper lambda to apply compute_grad_matrix to each matrix and return a tuple of results
    auto tuple_of_gradients = [&](auto&... mats) {
        return std::make_tuple(compute_grad_matrix(mats)...);
    };
    
    return std::apply(tuple_of_gradients, std::tie(matrices...));
}

template <typename Scalar, typename... Matrices>
Eigen::SparseMatrix<Scalar> hessian(
  const Var<Var<Scalar>> & f,
  Matrices &... matrices)
{
}
