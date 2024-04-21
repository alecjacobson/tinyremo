// Copyright (C) 2024 Alec Jacobson <alecjacobson@gmail.com>
#pragma once
#include <cmath>
#include <memory>
#include <vector>
#include <array>
#include <cassert>
#include <map>
#include <unordered_map>

// Adapted from the rust implementation at https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
namespace tinyremo
{
  template<typename Key, typename Value> using index_map = std::unordered_map<Key,Value>;

  template<typename Scalar>
  struct Node 
  {
    std::array<Scalar, 2> weights;
    std::array<size_t, 2> deps;
  };

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
    // `Eigen::Matrix<Var<Scalar>,…>` will call `Var<Scalar>(0)` so we need a
    // construct that takes a literal.
    //
    // Since won't know the tape, so we can never follow grads through these,
    // but we claim this is ok because they are always constants.
    //
    // Eigen also calls math operators on constants:
    // `Var<Scalar>(0)/Var<Scalar>(1)` kind of thing. So we need to support
    // those below (some are asserted to be unsupported currently).
    Var(Scalar value = Scalar(0)): tape_ptr(nullptr), index(0), value(value) { }

    // Constructor for scalar types (e.g., int, double), but not for Var types.
    // SFINAE is used to enable this constructor only for scalar types.
    template<typename T, typename = typename std::enable_if<std::is_scalar<T>::value>::type>
    constexpr explicit Var(T v) : tape_ptr(nullptr), index(0), value(v) { }

    // Construct for a tracked variable
    Var(Tape<Scalar>* tape_ptr, size_t index, const Scalar & value) : tape_ptr(tape_ptr), index(index), value(value) {}

    Var operator+(const Var& other) const
    {
      assert(!tape_ptr || !other.tape_ptr || tape_ptr == other.tape_ptr);
      if(tape_ptr && other.tape_ptr) { return Var(tape_ptr, tape_ptr->push_binary(index, Scalar(1), other.index, Scalar(1)), value + other.value); }
      if(tape_ptr) { return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(1)), value + other.value); }
      if(other.tape_ptr) { return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, Scalar(1)), value + other.value); }
      return Var(value + other.value);
    }

    Var operator-(const Var& other) const
    {
      return *this + (-other);
    }

    Var operator*(const Var& other) const
    {
      assert(!tape_ptr || !other.tape_ptr || tape_ptr == other.tape_ptr);
      if(tape_ptr && other.tape_ptr) { return Var(tape_ptr, tape_ptr->push_binary(index, other.value, other.index, value), value * other.value); }
      if(tape_ptr) { return Var(tape_ptr, tape_ptr->push_unary(index, other.value), value * other.value); }
      if(other.tape_ptr) { return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, value), value * other.value); }
      return Var(value * other.value);
    }

    Var operator/(const Var& other) const
    {
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
        return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, Scalar(1) / (Scalar(2)*sqrt(v.value))), sqrt(v.value));
      }else
      {
        return Var(sqrt(v.value));
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

    index_map<size_t, Scalar> sparse_grad() const 
    {
      index_map<size_t, Scalar> derivs;
      compute_sparse_grad(derivs, Scalar(1), index);
      return derivs;
    }

    Scalar getValue() const { return value; }
    size_t getIndex() const { return index; }

  private:
    void compute_sparse_grad(index_map<size_t, Scalar>& derivs, Scalar grad_value, size_t idx) const
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

  // ```
  // Tape<double> tape_1;
  // Tape<Var<double>> tape_2;
  // Var<Var<double>> x = record_scalar<double>(0.5,tape_1,tape_2);
  // ```
  template<typename Scalar, typename FirstTape, typename... RestTapes>
  auto record_scalar(Scalar value, FirstTape& first_tape, RestTapes&... rest_tapes)
  {
    Var<Scalar> x(&first_tape, first_tape.push_scalar(), value);
    if constexpr (sizeof...(RestTapes) == 0) { return x; }
    else { return record_scalar(x, rest_tapes...); }
  }
}
