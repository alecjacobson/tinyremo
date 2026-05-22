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

  // Ordered worklist for sparse_grad.  Stays as a sorted flat-vector while
  // the active set is small (cache-friendly, O(1) pop_max, O(k) insert).
  // Once it exceeds FlatCapacity it spills to std::map (O(log k) insert and
  // pop_max), avoiding the O(k) insert cost that causes O(n²) behaviour when
  // a dense row forces the worklist to grow large.
  template<typename Key, typename Value,
           size_t FlatCapacity = 32, typename Cmp = std::less<Key>>
  struct worklist_map {
    using Entry = std::pair<Key, Value>;

    bool empty() const { return spilled ? tree.empty() : flat.empty(); }

    Value& operator[](const Key& k) {
      if (spilled) return tree[k];
      auto it = flat_lb(k);
      if (it != flat.end() && !cmp(k, it->first) && !cmp(it->first, k))
        return it->second;
      if (flat.size() < FlatCapacity)
        return flat.insert(it, {k, Value{}})->second;
      spill();
      return tree[k];
    }

    // Pop the maximum-key entry.  O(1) in flat mode, O(log k) in tree mode.
    Entry pop_max() {
      if (spilled) {
        auto it = std::prev(tree.end());
        Entry e = *it; tree.erase(it); return e;
      }
      Entry e = flat.back(); flat.pop_back(); return e;
    }

    void emplace(Key k, Value val) { (*this)[k] = std::move(val); }

  private:
    std::vector<Entry>      flat;
    std::map<Key,Value,Cmp> tree;
    bool spilled = false;
    Cmp  cmp{};

    auto flat_lb(const Key& k) {
      return std::lower_bound(flat.begin(), flat.end(), k,
        [this](const Entry& a, const Key& b){ return cmp(a.first, b); });
    }

    void spill() {
      for (auto& [k,v] : flat) tree.emplace(std::move(k), std::move(v));
      flat.clear(); flat.shrink_to_fit(); spilled = true;
    }
  };

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
    void   reserve(size_t n) { nodes.reserve(n); }

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
      // x + 0 = x, 0 + x = x: skip unary wrapper nodes (e.g. in backward sweeps).
      if( tape_ptr && !other.tape_ptr && other.value == Scalar(0)) return *this;
      if(!tape_ptr &&  other.tape_ptr &&       value == Scalar(0)) return other;
      if(tape_ptr && other.tape_ptr) {
        // x + x: fold into a single unary node with weight 2 instead of binary.
        if(index == other.index) return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(2)), value + other.value);
        return Var(tape_ptr, tape_ptr->push_binary(index, Scalar(1), other.index, Scalar(1)), value + other.value);
      }
      if(tape_ptr) { return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(1)), value + other.value); }
      if(other.tape_ptr) { return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, Scalar(1)), value + other.value); }
      return Var(value + other.value);
    }

    Var operator-(const Var& other) const
    {
      assert(!tape_ptr || !other.tape_ptr || tape_ptr == other.tape_ptr);
      // x - 0 = x: skip unary wrapper node.
      if(tape_ptr && !other.tape_ptr && other.value == Scalar(0)) return *this;
      if(tape_ptr && other.tape_ptr) { return Var(tape_ptr, tape_ptr->push_binary(index, Scalar(1), other.index, Scalar(-1)), value - other.value); }
      if(tape_ptr) { return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(1)), value - other.value); }
      if(other.tape_ptr) { return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, Scalar(-1)), value - other.value); }
      return Var(value - other.value);
    }

    Var operator*(const Var& other) const
    {
      assert(!tape_ptr || !other.tape_ptr || tape_ptr == other.tape_ptr);
      // x * 1 = x, 1 * x = x: skip unary wrapper nodes (e.g. in backward sweeps).
      if( tape_ptr && !other.tape_ptr && other.value == Scalar(1)) return *this;
      if(!tape_ptr &&  other.tape_ptr &&       value == Scalar(1)) return other;
      // x * 0 = 0, 0 * x = 0: prune zero branches (e.g. leaf self-loop nodes).
      // Only safe when the zero has no inner tape (pure constant, not a tracked zero).
      if( tape_ptr && !other.tape_ptr && other.value == Scalar(0)) return Var(Scalar(0));
      if(!tape_ptr &&  other.tape_ptr &&       value == Scalar(0)) return Var(Scalar(0));
      if(tape_ptr && other.tape_ptr) {
        // x * x: fold into a single unary node with weight 2x instead of binary.
        if(index == other.index) return Var(tape_ptr, tape_ptr->push_unary(index, Scalar(2)*value), value * other.value);
        return Var(tape_ptr, tape_ptr->push_binary(index, other.value, other.index, value), value * other.value);
      }
      if(tape_ptr) { return Var(tape_ptr, tape_ptr->push_unary(index, other.value), value * other.value); }
      if(other.tape_ptr) { return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, value), value * other.value); }
      return Var(value * other.value);
    }

    Var operator/(const Var& other) const
    {
      assert(other.value != Scalar(0));
      assert(!tape_ptr || !other.tape_ptr || tape_ptr == other.tape_ptr);
      // x / 1 = x: skip unary wrapper node.
      if(tape_ptr && !other.tape_ptr && other.value == Scalar(1)) return *this;
      const Scalar inv  = Scalar(1) / other.value;          // 1/y
      const Scalar ninv2 = -value * inv * inv;              // -x/y²
      if(tape_ptr && other.tape_ptr) { return Var(tape_ptr, tape_ptr->push_binary(index, inv, other.index, ninv2), value * inv); }
      if(tape_ptr) { return Var(tape_ptr, tape_ptr->push_unary(index, inv), value * inv); }
      if(other.tape_ptr) { return Var(other.tape_ptr, other.tape_ptr->push_unary(other.index, ninv2), value * inv); }
      return Var(value * inv);
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

    friend Var sqrt(const Var& v)
    {
      using std::sqrt;
      if(v.tape_ptr)
      {
        const auto sv = sqrt(v.value);
        return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, Scalar(1) / (Scalar(2)*sv)), sv);
      }else
      {
        return Var(sqrt(v.value));
      }
    }
    friend Var sin(const Var& v)
    {
      using std::cos; using std::sin;
      if(v.tape_ptr)
      {
        const auto sv = sin(v.value), cv = cos(v.value);
        return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, cv), sv);
      }else
      {
        return Var(sin(v.value));
      }
    }
    friend Var cos(const Var& v)
    {
      using std::cos; using std::sin;
      if(v.tape_ptr)
      {
        const auto sv = sin(v.value), cv = cos(v.value);
        return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, -sv), cv);
      }else
      {
        return Var(cos(v.value));
      }
    }
    friend Var exp(const Var& v)
    {
      using std::exp;
      if(v.tape_ptr)
      {
        const auto ev = exp(v.value);
        return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, ev), ev);
      }else
      {
        return Var(exp(v.value));
      }
    }
    friend Var log(const Var& v)
    {
      using std::log;
      if(v.tape_ptr)
      {
        const auto lv = log(v.value);
        return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, Scalar(1) / v.value), lv);
      }else
      {
        return Var(log(v.value));
      }
    }
    friend Var pow(const Var& x, const Var& p)
    {
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
    friend Var abs(const Var& v)
    {
      using std::abs;
      if(v.tape_ptr)
      {
        return Var(v.tape_ptr, v.tape_ptr->push_unary(v.index, v.value < Scalar(0) ? Scalar(-1) : Scalar(1)), abs(v.value));
      }else
      {
        return Var(abs(v.value));
      }
    }

    std::vector<Scalar> grad() const
    {
      std::vector<Scalar> derivs;
      grad(derivs);
      return derivs;
    }

    // In-place variant: reuses a caller-supplied buffer, avoiding
    // repeated heap allocation when grad() is called in a tight loop.
    void grad(std::vector<Scalar>& out) const
    {
      out.assign(tape_ptr->size(), Scalar(0));
      out[index] = Scalar(1);
      for (int i = (int)tape_ptr->size() - 1; i >= 0; --i) {
        // Zero-skip is only safe for plain arithmetic scalars.  When
        // Scalar = Var<T> (nested tape), value == 0 does NOT imply that
        // the inner derivative is zero, so skipping would silently drop
        // second-derivative contributions (e.g. Hessian at a gradient
        // minimum where all first-order adjoints are zero but H ≠ 0).
        if constexpr (std::is_arithmetic_v<Scalar>) {
          if (out[i] == Scalar(0)) continue;
        }
        const auto& node = (*tape_ptr)[i];
        for (int j = 0; j < 2; ++j) {
          if (node.deps[j] == (size_t)i) continue;  // self-loop (leaf/unary slot)
          out[node.deps[j]] += node.weights[j] * out[i];
        }
      }
    }

    index_map<size_t, Scalar> sparse_grad() const
    {
      // Sparse iterative reverse sweep.  worklist_map: flat sorted-vector
      // while the active set is small, spills to std::map beyond FlatCapacity.
      worklist_map<size_t, Scalar> wl;
      wl[index] = Scalar(1);

      index_map<size_t, Scalar> result;
      while (!wl.empty()) {
        auto [i, g] = wl.pop_max();

        // Same guard as grad(): when Scalar=Var<T> a zero primal adjoint still
        // carries inner-tape second-derivative info, so we must never skip it.
        if constexpr (std::is_arithmetic_v<Scalar>) {
          if (g == Scalar(0)) continue;
        }
        result.emplace(i, g);

        const auto& node = (*tape_ptr)[i];
        for (int j = 0; j < 2; ++j) {
          const size_t dep = node.deps[j];
          if (dep < i) wl[dep] += node.weights[j] * g;
        }
      }
      return result;
    }

    Scalar getValue() const { return value; }
    size_t getIndex() const { return index; }

  private:
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
