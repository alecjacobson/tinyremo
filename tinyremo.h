// Copyright (C) 2024 Alec Jacobson <alecjacobson@gmail.com>
#include <cmath>
#include <memory>
#include <vector>
#include <array>
#include <cassert>
#include <map>

// Adapted from the rust implementation at https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
template<typename Scalar>
struct Node 
{
  std::array<Scalar, 2> weights;
  std::array<size_t, 2> deps;
};

// Forward declaration
template<typename Scalar> class Var;

template<typename Scalar>
class Tape 
{
public:
  size_t push_scalar()
  {
    size_t index = nodes.size();
    nodes.push_back({{Scalar(0),Scalar(0)}, {index, index}});
    return index;
  }

  size_t push_unary(size_t dep0, Scalar weight0)
  {
    size_t index = nodes.size();
    nodes.push_back({{weight0, Scalar(0)}, {dep0, index}});
    return index;
  }

  size_t push_binary(size_t dep0, Scalar weight0, size_t dep1, Scalar weight1)
  {
    size_t index = nodes.size();
    nodes.push_back({{weight0, weight1}, {dep0, dep1}});
    return index;
  }

  size_t size() const { return nodes.size(); }

  const Node<Scalar>& operator[](size_t index) const { return nodes[index]; }

private:
  std::vector<Node<Scalar>> nodes;
};


template<typename Scalar>
class Var {
public:
  // `Eigen::Matrix<Var<Scalar>,3,2> x;` will invoke the default constructor.
  //
  // We could combine this with the next constructor `Var(Scalar value =
  // Scalar(0))`.
  Var(): tape_ptr(nullptr), index(0), value(0) { }

  // `Eigen::Matrix<Var<Scalar>,…>` will call `Var<Scalar>(0)` so we need a
  // construct that takes a literal.
  //
  // Since won't know the tape, so we can never follow grads through these,
  // but we claim this is ok because they are always constants.
  //
  // Eigen also calls math operators on constants:
  // `Var<Scalar>(0)/Var<Scalar>(1)` kind of thing. So we need to support
  // those below (some are asserted to be unsupported currently).
  Var(Scalar value): tape_ptr(nullptr), index(0), value(value) { }

  // Construct for a tracked variable
  Var(Tape<Scalar>* tape_ptr, size_t index, const Scalar & value) :
    tape_ptr(tape_ptr), index(index), value(value) {}

  Var operator+(const Var& other) const
  {
    if(tape_ptr)
    {
      if(other.tape_ptr)
      {
        assert(tape_ptr == other.tape_ptr);
        return Var(tape_ptr, tape_ptr->push_binary(index, Scalar(1), other.index, Scalar(1)), value + other.value);
      }else
      {
        return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(1)), value + other.value);
      }
    }else
    {
      if(other.tape_ptr)
      {
        return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, Scalar(1)), value + other.value);
      }else
      {
        return Var(value + other.value);
      }
    }
  }

  Var operator-(const Var& other) const
  {
    assert(tape_ptr == other.tape_ptr);
    return *this + (-other);
  }

  Var operator*(const Var& other) const
  {
    if(tape_ptr)
    {
      if(other.tape_ptr)
      {
        assert(tape_ptr == other.tape_ptr);
        return Var(tape_ptr, tape_ptr->push_binary(index, other.value, other.index, value), value * other.value);
      }else
      {
        return Var(tape_ptr, tape_ptr->push_unary(index, other.value), value * other.value);
      }
    }else
    {
      if(other.tape_ptr)
      {
        return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, value), value * other.value);
      }else
      {
        return Var(value * other.value);
      }
    }
  }

  Var operator/(const Var& other) const
  {
    assert(tape_ptr == other.tape_ptr);
    assert(other.value != Scalar(0));
    return *this * other.invert();
  }

  Var invert() const
  {
    if(tape_ptr)
    {
      return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(-1) / (value * value)), Scalar(1) / value);
    }else
    {
      return Var(Scalar(1) / value);
    }
  }


  // Unary negation
  Var operator-() const
  {
    if(tape_ptr)
    {
      return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(-1)), -value);
    }else
    {
      return Var(-value);
    }
  }

  // -= operator
  Var& operator-=(const Var& other) { *this = *this - other; return *this; }
  Var& operator+=(const Var& other) { *this = *this + other; return *this; }
  Var& operator*=(const Var& other) { *this = *this * other; return *this; }
  Var& operator/=(const Var& other) { *this = *this / other; return *this; }


  // > operator
  bool operator>(const Var& other) const { return value > other.value; }
  bool operator>=(const Var& other) const { return value >= other.value; }
  bool operator<(const Var& other) const { return value < other.value; }
  bool operator<=(const Var& other) const { return value <= other.value; }
  bool operator==(const Var& other) const { return value == other.value; }


  // Eigen will call `sqrt(x(i))` for `Eigen::Matrix<Var,…> x;`
  friend Var sqrt(const Var& v) {
    using std::sqrt;
    if(v.tape_ptr)
    {
      return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, Scalar(1.0) / (Scalar(2)*sqrt(v.value))), sqrt(v.value));
    }else
    {
      return Var(std::sqrt(v.value));
    }
  }
  friend Var sin(const Var& v) {
    using std::cos;
    using std::sin;
    if(v.tape_ptr)
    {
      return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, cos(v.value)), sin(v.value));
    }else
    {
      return Var(sin(v.value));
    }
  }
  friend Var cos(const Var& v) {
    using std::cos;
    using std::sin;
    if(v.tape_ptr)
    {
      return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, -sin(v.value)), cos(v.value));
    }else
    {
      return Var(cos(v.value));
    }
  }
  friend Var exp(const Var& v) {
    using std::exp;
    if(v.tape_ptr)
    {
      return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, exp(v.value)), exp(v.value));
    }else
    {
      return Var(exp(v.value));
    }
  }
  friend Var log(const Var& v) {
    using std::log;
    if(v.tape_ptr)
    {
      return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, Scalar(1) / v.value), log(v.value));
    }else
    {
      return Var(log(v.value));
    }
  }
  friend Var pow(const Var& x, const Var& p) {
    using std::pow;
    // Mirror operator+ above
    if(x.tape_ptr)
    {
      if(p.tape_ptr)
      {
        assert(x.tape_ptr == p.tape_ptr);
        return Var(x.tape_ptr, x.tape_ptr->push_binary(x.index, p.value * pow(x.value, p.value - Scalar(1)), p.index, pow(x.value, p.value) * log(x.value)), pow(x.value, p.value));
      }else
      {
        return Var(x.tape_ptr, x.tape_ptr->push_unary(x.index, p.value * pow(x.value, p.value - Scalar(1))), pow(x.value, p.value));
      }
    }else
    {
      if(p.tape_ptr)
      {
        return Var(p.tape_ptr, p.tape_ptr->push_unary(p.index, pow(x.value, p.value) * log(x.value)), pow(x.value, p.value));
      }else
      {
        return Var(pow(x.value, p.value));
      }
    }
  }

  std::vector<Scalar> grad() const
  {
    std::vector<Scalar> derivs(tape_ptr->size(), Scalar(0));
    derivs[index] = Scalar(1);

    for (int i = tape_ptr->size() - 1; i >= 0; --i) {
      const auto& node = (*tape_ptr)[i];
      for (int j = 0; j < 2; ++j) {
        derivs[node.deps[j]] += node.weights[j] * derivs[i];
      }
    }
    return derivs;
  }

  // Could use std::unordered_map if we decide we don't care about order here.
  std::map<size_t, Scalar> sparse_grad() const 
  {
    std::map<size_t, Scalar> derivs;
    compute_sparse_grad(derivs, Scalar(1), index);
    return derivs;
  }

  Scalar getValue() const { return value; }
  size_t getIndex() const { return index; }

private:
  void compute_sparse_grad(std::map<size_t, Scalar>& derivs, Scalar grad_value, size_t idx) const
  {
    assert(idx < tape_ptr->size() && idx >= 0);
    if (derivs.find(idx) != derivs.end()) {
      derivs[idx] += grad_value;
    } else {
      derivs[idx] = grad_value;
    }
    const Node<Scalar>& node = (*tape_ptr)[idx];
    for (int i = 0; i < 2; ++i) {
      size_t dep_idx = node.deps[i];
      if (dep_idx < idx) {
        Scalar weight = node.weights[i];
        compute_sparse_grad(derivs, grad_value * weight, dep_idx);
      }
    }
  }

  Tape<Scalar> * tape_ptr;
  size_t index;
  Scalar value;
};

