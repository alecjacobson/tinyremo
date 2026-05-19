#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "tinyremo_eigen.h"
#include <Eigen/Dense>
#include <cmath>
#include <numbers>

using namespace tinyremo;
using V1 = Var<double>;
using V2 = Var<V1>;

static const double TOL = 1e-10;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Create a first-order tracked variable
static V1 var(double v, Tape<double>& t)
{
    return V1(&t, t.push_scalar(), v);
}

// df/dx, both on the same tape
static double grad1(V1 f, V1 x)
{
    return f.grad()[x.getIndex()];
}

// d²f/(d wrt_outer)(d wrt_inner)
//   outer: differentiate f w.r.t. wrt_outer in the outer (tape2) reverse pass
//   inner: differentiate that result w.r.t. wrt_inner in the inner (tape1) reverse pass
static double d2(V2 f, V2 wrt_outer, V2 wrt_inner)
{
    auto outer = f.grad();                             // vector<V1>
    V1   df    = outer[wrt_outer.getIndex()];
    auto inner = df.grad();                            // vector<double>
    return inner[wrt_inner.getValue().getIndex()];
}

// ─────────────────────────────────────────────────────────────────────────────
// Tape internals
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Tape: push_scalar creates self-loop node with zero weights")
{
    Tape<double> t;
    size_t i = t.push_scalar();
    CHECK(i == 0);
    CHECK(t.size() == 1);
    CHECK(t[i].deps[0] == i);
    CHECK(t[i].deps[1] == i);
    CHECK(t[i].weights[0] == 0.0);
    CHECK(t[i].weights[1] == 0.0);
}

TEST_CASE("Tape: push_unary stores dep and weight, second dep is self")
{
    Tape<double> t;
    size_t i = t.push_scalar();
    size_t j = t.push_unary(i, 3.0);
    CHECK(j == 1);
    CHECK(t[j].deps[0]   == i);
    CHECK(t[j].weights[0] == 3.0);
    CHECK(t[j].deps[1]   == j);   // self-loop on unused slot
    CHECK(t[j].weights[1] == 0.0);
}

TEST_CASE("Tape: push_binary stores two deps and weights")
{
    Tape<double> t;
    size_t a = t.push_scalar();
    size_t b = t.push_scalar();
    size_t c = t.push_binary(a, 2.0, b, 5.0);
    CHECK(c == 2);
    CHECK(t[c].deps[0]   == a);
    CHECK(t[c].deps[1]   == b);
    CHECK(t[c].weights[0] == 2.0);
    CHECK(t[c].weights[1] == 5.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Primal values
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Var: primal values for all ops")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t);

    CHECK((x + y).getValue() == doctest::Approx(5.0));
    CHECK((x - y).getValue() == doctest::Approx(-1.0));
    CHECK((x * y).getValue() == doctest::Approx(6.0));
    CHECK((x / y).getValue() == doctest::Approx(2.0/3.0));
    CHECK((-x).getValue()    == doctest::Approx(-2.0));
    CHECK(sqrt(x).getValue() == doctest::Approx(std::sqrt(2.0)));
    CHECK(sin(x).getValue()  == doctest::Approx(std::sin(2.0)));
    CHECK(cos(x).getValue()  == doctest::Approx(std::cos(2.0)));
    CHECK(exp(x).getValue()  == doctest::Approx(std::exp(2.0)));
    CHECK(log(x).getValue()  == doctest::Approx(std::log(2.0)));
    CHECK(pow(x,y).getValue() == doctest::Approx(8.0));
    CHECK(abs(x).getValue()  == doctest::Approx(2.0));
    CHECK(abs(-x).getValue() == doctest::Approx(2.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// First derivatives
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("grad: addition df/dx=1, df/dy=1")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t);
    auto g = (x + y).grad();
    CHECK(g[x.getIndex()] == doctest::Approx(1.0));
    CHECK(g[y.getIndex()] == doctest::Approx(1.0));
}

TEST_CASE("grad: subtraction df/dx=1, df/dy=-1")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t);
    auto g = (x - y).grad();
    CHECK(g[x.getIndex()] == doctest::Approx( 1.0));
    CHECK(g[y.getIndex()] == doctest::Approx(-1.0));
}

TEST_CASE("grad: multiplication df/dx=y, df/dy=x")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t);
    auto g = (x * y).grad();
    CHECK(g[x.getIndex()] == doctest::Approx(3.0));
    CHECK(g[y.getIndex()] == doctest::Approx(2.0));
}

TEST_CASE("grad: division df/dx=1/y, df/dy=-x/y^2")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t);
    auto g = (x / y).grad();
    CHECK(g[x.getIndex()] == doctest::Approx(1.0/3.0));
    CHECK(g[y.getIndex()] == doctest::Approx(-2.0/9.0));
}

TEST_CASE("grad: negation df/dx=-1")
{
    Tape<double> t;
    V1 x = var(2.0, t);
    CHECK(grad1(-x, x) == doctest::Approx(-1.0));
}

TEST_CASE("grad: sqrt  df/dx = 1/(2*sqrt(x))")
{
    Tape<double> t;
    V1 x = var(4.0, t);
    CHECK(grad1(sqrt(x), x) == doctest::Approx(0.25));
}

TEST_CASE("grad: sin  df/dx = cos(x)")
{
    Tape<double> t;
    V1 x = var(1.0, t);
    CHECK(grad1(sin(x), x) == doctest::Approx(std::cos(1.0)));
}

TEST_CASE("grad: cos  df/dx = -sin(x)")
{
    Tape<double> t;
    V1 x = var(1.0, t);
    CHECK(grad1(cos(x), x) == doctest::Approx(-std::sin(1.0)));
}

TEST_CASE("grad: exp  df/dx = exp(x)")
{
    Tape<double> t;
    V1 x = var(1.0, t);
    CHECK(grad1(exp(x), x) == doctest::Approx(std::exp(1.0)));
}

TEST_CASE("grad: log  df/dx = 1/x")
{
    Tape<double> t;
    V1 x = var(2.0, t);
    CHECK(grad1(log(x), x) == doctest::Approx(0.5));
}

TEST_CASE("grad: pow(x, constant p)  df/dx = p*x^(p-1)")
{
    Tape<double> t;
    V1 x = var(2.0, t);
    V1 p(3.0);                      // untracked constant
    CHECK(grad1(pow(x, p), x) == doctest::Approx(12.0));
}

TEST_CASE("grad: pow(constant c, x)  df/dx = c^x * ln(c)")
{
    Tape<double> t;
    V1 c(2.0);                      // untracked
    V1 x = var(3.0, t);
    double expected = std::pow(2.0, 3.0) * std::log(2.0);
    CHECK(grad1(pow(c, x), x) == doctest::Approx(expected));
}

TEST_CASE("grad: pow(x,y) both tracked")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t);
    auto g = pow(x, y).grad();
    CHECK(g[x.getIndex()] == doctest::Approx(3.0 * 4.0));                          // p*x^(p-1)
    CHECK(g[y.getIndex()] == doctest::Approx(8.0 * std::log(2.0)));                // x^p*ln(x)
}

TEST_CASE("grad: abs positive x → +1, negative x → -1")
{
    {
        Tape<double> t;
        V1 x = var(2.0, t);
        CHECK(grad1(abs(x), x) == doctest::Approx(1.0));
    }
    {
        Tape<double> t;
        V1 x = var(-2.0, t);
        CHECK(grad1(abs(x), x) == doctest::Approx(-1.0));
    }
}

TEST_CASE("grad: chain rule  d/dx sin(x^2) = 2x*cos(x^2)")
{
    Tape<double> t;
    V1 x = var(1.5, t);
    double expected = 2.0 * 1.5 * std::cos(1.5 * 1.5);
    CHECK(grad1(sin(x * x), x) == doctest::Approx(expected));
}

TEST_CASE("grad: multi-variable  f = x^2 + x*y + y^2")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t);
    auto g = (x*x + x*y + y*y).grad();
    CHECK(g[x.getIndex()] == doctest::Approx(7.0));  // 2x+y
    CHECK(g[y.getIndex()] == doctest::Approx(8.0));  // x+2y
}

TEST_CASE("grad: untracked constant adds unary (not binary) node")
{
    Tape<double> t;
    V1 x = var(2.0, t);
    size_t before = t.size();
    V1 f = V1(5.0) * x;             // constant * tracked
    CHECK(t.size() == before + 1);  // one unary node, not binary
    CHECK(grad1(f, x) == doctest::Approx(5.0));
}

TEST_CASE("grad: unused variable has zero derivative")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t), z = var(4.0, t);
    auto g = (x * y).grad();
    CHECK(g[z.getIndex()] == doctest::Approx(0.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// grad vs sparse_grad consistency
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("sparse_grad: every entry matches dense grad")
{
    Tape<double> t;
    V1 x = var(1.5, t), y = var(2.5, t);
    V1 f = sin(x*x) + log(y) + x*y;
    auto g_dense  = f.grad();
    auto g_sparse = f.sparse_grad();
    for (auto& [idx, val] : g_sparse)
        CHECK(val == doctest::Approx(g_dense[idx]).epsilon(1e-14));
}

TEST_CASE("sparse_grad: variable not in expression is absent or zero")
{
    Tape<double> t;
    V1 x = var(2.0, t), y = var(3.0, t), z = var(4.0, t);
    auto g = (x * y).sparse_grad();
    auto it = g.find(z.getIndex());
    CHECK((it == g.end() || it->second == doctest::Approx(0.0)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Second derivatives via nested tapes
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("hessian scalar: x^2  →  d²f/dx² = 2")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(2.0, t1, t2);
    CHECK(d2(x*x, x, x) == doctest::Approx(2.0));
}

TEST_CASE("hessian scalar: x^3 at x=2  →  d²f/dx² = 12")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(2.0, t1, t2);
    CHECK(d2(x*x*x, x, x) == doctest::Approx(12.0));
}

TEST_CASE("hessian scalar: sin(x) at x=1  →  d²f/dx² = -sin(1)")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(1.0, t1, t2);
    CHECK(d2(sin(x), x, x) == doctest::Approx(-std::sin(1.0)));
}

TEST_CASE("hessian scalar: cos(x) at x=1  →  d²f/dx² = -cos(1)")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(1.0, t1, t2);
    CHECK(d2(cos(x), x, x) == doctest::Approx(-std::cos(1.0)));
}

TEST_CASE("hessian scalar: exp(x) at x=1  →  d²f/dx² = e")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(1.0, t1, t2);
    CHECK(d2(exp(x), x, x) == doctest::Approx(std::exp(1.0)));
}

TEST_CASE("hessian scalar: log(x) at x=2  →  d²f/dx² = -1/4")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(2.0, t1, t2);
    CHECK(d2(log(x), x, x) == doctest::Approx(-0.25));
}

TEST_CASE("hessian scalar: sqrt(x) at x=4  →  d²f/dx² = -1/32")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(4.0, t1, t2);
    CHECK(d2(sqrt(x), x, x) == doctest::Approx(-1.0/32.0));
}

TEST_CASE("hessian scalar: x*y  →  diagonal 0, cross = 1")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(2.0, t1, t2);
    V2 y = record_scalar(3.0, t1, t2);
    V2 f = x * y;
    CHECK(std::abs(d2(f, x, x)) < TOL);
    CHECK(std::abs(d2(f, y, y)) < TOL);
    CHECK(d2(f, x, y) == doctest::Approx(1.0));
    CHECK(d2(f, y, x) == doctest::Approx(1.0));
}

TEST_CASE("hessian scalar: x/y at (2,3)  →  d²/dxdy=-1/9, d²/dy²=4/27")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(2.0, t1, t2);
    V2 y = record_scalar(3.0, t1, t2);
    V2 f = x / y;
    CHECK(std::abs(d2(f, x, x)) < TOL);
    CHECK(d2(f, x, y) == doctest::Approx(-1.0/9.0));
    CHECK(d2(f, y, y) == doctest::Approx(4.0/27.0));
}

TEST_CASE("hessian scalar: x^2*y at (2,3)  →  d²/dx²=6, d²/dxdy=4, d²/dy²=0")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(2.0, t1, t2);
    V2 y = record_scalar(3.0, t1, t2);
    V2 f = x*x*y;
    CHECK(d2(f, x, x) == doctest::Approx(6.0));   // 2y
    CHECK(d2(f, x, y) == doctest::Approx(4.0));   // 2x
    CHECK(std::abs(d2(f, y, y)) < TOL);
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigen: gradient()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("eigen gradient: f(X)=X.sum() → all-ones")
{
    Tape<double> t;
    Eigen::Matrix<double,2,3> Xd; Xd << 1,2,3,4,5,6;
    auto X = record_matrix(Xd, t);
    auto [dX] = gradient(X.sum(), X);
    CHECK((dX - Eigen::Matrix<double,2,3>::Ones()).norm() < TOL);
}

TEST_CASE("eigen gradient: f(X)=||X||^2 → 2X")
{
    Tape<double> t;
    Eigen::Matrix<double,2,2> Xd; Xd << 1,2,3,4;
    auto X = record_matrix(Xd, t);
    auto [dX] = gradient(X.squaredNorm(), X);
    CHECK((dX - 2.0*Xd).norm() < TOL);
}

TEST_CASE("eigen gradient: f(x,y)=x.dot(y) → dfdx=y, dfdy=x")
{
    Tape<double> t;
    Eigen::Vector3d xd(1,2,3), yd(4,5,6);
    auto x = record_matrix(xd, t);
    auto y = record_matrix(yd, t);
    auto [dx, dy] = gradient(x.dot(y), x, y);
    CHECK((dx - yd).norm() < TOL);
    CHECK((dy - xd).norm() < TOL);
}

TEST_CASE("eigen gradient: preserves matrix shape (3x2)")
{
    Tape<double> t;
    Eigen::Matrix<double,3,2> Xd = Eigen::Matrix<double,3,2>::Random();
    auto X = record_matrix(Xd, t);
    auto [dX] = gradient(X.squaredNorm(), X);
    CHECK(dX.rows() == 3);
    CHECK(dX.cols() == 2);
}

TEST_CASE("eigen gradient: dynamic matrix")
{
    Tape<double> t;
    Eigen::MatrixXd Xd = Eigen::MatrixXd::Ones(4,3);
    auto X = record_matrix(Xd, t);
    auto [dX] = gradient(X.sum(), X);
    CHECK((dX - Eigen::MatrixXd::Ones(4,3)).norm() < TOL);
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigen: dense hessian()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("dense hessian: 1x1 fixed-size  f=x^2 → H=[[2]]")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Matrix<double,1,1> xd; xd << 3.0;
    auto X = record_matrix(xd, t1, t2);
    auto H = hessian(X(0,0)*X(0,0), X);
    CHECK(H.rows() == 1);
    CHECK(H.cols() == 1);
    CHECK(H(0,0) == doctest::Approx(2.0));
}

TEST_CASE("dense hessian: f(x,y)=x^2+2y^2 → diag(2,4)")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Vector2d xd(1.5, 2.5);
    auto X = record_matrix(xd, t1, t2);
    auto f = X(0)*X(0) + V2(2)*X(1)*X(1);
    auto H = hessian(f, X);
    CHECK(H(0,0) == doctest::Approx(2.0));
    CHECK(H(1,1) == doctest::Approx(4.0));
    CHECK(std::abs(H(0,1)) < TOL);
    CHECK(std::abs(H(1,0)) < TOL);
}

TEST_CASE("dense hessian: f(x,y)=x*y → [[0,1],[1,0]]")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Vector2d xd(1.0, 2.0);
    auto X = record_matrix(xd, t1, t2);
    auto H = hessian(X(0)*X(1), X);
    CHECK(std::abs(H(0,0)) < TOL);
    CHECK(std::abs(H(1,1)) < TOL);
    CHECK(H(0,1) == doctest::Approx(1.0));
    CHECK(H(1,0) == doctest::Approx(1.0));
}

TEST_CASE("dense hessian: f(v)=||v||^2 → 2*I  (dynamic vector)")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::VectorXd vd = Eigen::VectorXd::LinSpaced(4, 1.0, 4.0);
    auto V = record_matrix(vd, t1, t2);
    auto H = hessian(V.squaredNorm(), V);
    CHECK(H.rows() == 4);
    CHECK(H.cols() == 4);
    CHECK((H - 2.0*Eigen::MatrixXd::Identity(4,4)).norm() < TOL);
}

TEST_CASE("dense hessian: row-major input gives correct values")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Matrix<double,1,2,Eigen::RowMajor> xd; xd << 3.0, 4.0;
    auto X = record_matrix(xd, t1, t2);
    auto H = hessian(X(0,0)*X(0,0) + X(0,1)*X(0,1), X);
    CHECK(H(0,0) == doctest::Approx(2.0));
    CHECK(H(1,1) == doctest::Approx(2.0));
    CHECK(std::abs(H(0,1)) < TOL);
    CHECK(std::abs(H(1,0)) < TOL);
}

TEST_CASE("dense hessian: non-trivial 3-variable function is symmetric")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Vector3d xd(1.0, 2.0, 3.0);
    auto X = record_matrix(xd, t1, t2);
    auto f = sin(X(0)*X(1)) + exp(X(1)+X(2)) + X(0)*X(0)*X(2);
    auto H = hessian(f, X);
    CHECK((H - H.transpose()).norm() < TOL);
}

TEST_CASE("dense hessian: multiple matrix inputs")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Vector2d xd(1.0, 2.0), yd(3.0, 4.0);
    auto X = record_matrix(xd, t1, t2);
    auto Y = record_matrix(yd, t1, t2);
    // f = x.dot(y) = x0*y0 + x1*y1  →  H is 4x4 with cross-terms only
    auto f = X.dot(Y);
    auto H = hessian(f, X, Y);
    CHECK(H.rows() == 4);
    CHECK(H.cols() == 4);
    // All pure second derivatives are zero
    for (int i = 0; i < 4; i++)
        CHECK(std::abs(H(i,i)) < TOL);
    // dX(0) dY(0) = 1, etc.
    CHECK(H(0,2) == doctest::Approx(1.0));  // x0 wrt y0
    CHECK(H(1,3) == doctest::Approx(1.0));  // x1 wrt y1
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigen: sparse_hessian() – consistency with dense and size
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("sparse hessian: matches dense for f(x,y)=x^2+2y^2")
{
    Eigen::Vector2d xd(1.5, 2.5);

    Tape<double> td1; Tape<V1> td2;
    auto Xd = record_matrix(xd, td1, td2);
    auto Hd = hessian(Xd(0)*Xd(0) + V2(2)*Xd(1)*Xd(1), Xd);

    Tape<double> ts1; Tape<V1> ts2;
    auto Xs = record_matrix(xd, ts1, ts2);
    auto Hs = sparse_hessian(Xs(0)*Xs(0) + V2(2)*Xs(1)*Xs(1), Xs);

    CHECK((Eigen::MatrixXd(Hs) - Hd).norm() < TOL);
}

TEST_CASE("sparse hessian: matches dense for ||v||^2")
{
    Eigen::VectorXd vd = Eigen::VectorXd::LinSpaced(5, 1.0, 5.0);

    Tape<double> td1; Tape<V1> td2;
    auto Vd = record_matrix(vd, td1, td2);
    auto Hd = hessian(Vd.squaredNorm(), Vd);

    Tape<double> ts1; Tape<V1> ts2;
    auto Vs = record_matrix(vd, ts1, ts2);
    auto Hs = sparse_hessian(Vs.squaredNorm(), Vs);

    CHECK((Eigen::MatrixXd(Hs) - Hd).norm() < TOL);
}

TEST_CASE("sparse hessian: correct size NxN")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::VectorXd vd = Eigen::VectorXd::Ones(6);
    auto V = record_matrix(vd, t1, t2);
    auto H = sparse_hessian(V.squaredNorm(), V);
    CHECK(H.rows() == 6);
    CHECK(H.cols() == 6);
}

TEST_CASE("sparse hessian: f(x,y)=x*y has zero diagonal entries")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Vector2d xd(2.0, 3.0);
    auto X = record_matrix(xd, t1, t2);
    auto H = sparse_hessian(X(0)*X(1), X);
    CHECK(std::abs(H.coeff(0,0)) < TOL);
    CHECK(std::abs(H.coeff(1,1)) < TOL);
    CHECK(H.coeff(0,1) == doctest::Approx(1.0));
    CHECK(H.coeff(1,0) == doctest::Approx(1.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Eigen: sparse_jacobian()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("sparse jacobian: identity map F(x)=x → J=I")
{
    Tape<double> t;
    Eigen::Vector4d xd(1,2,3,4);
    auto X = record_matrix(xd, t);
    std::vector<V1> F{X(0), X(1), X(2), X(3)};
    auto J = sparse_jacobian<double>(F, X);
    CHECK(J.rows() == 4);
    CHECK(J.cols() == 4);
    CHECK((Eigen::MatrixXd(J) - Eigen::Matrix4d::Identity()).norm() < TOL);
}

TEST_CASE("sparse jacobian: F=[x0+x1, x1+x2, x2+x0]")
{
    Tape<double> t;
    Eigen::Vector3d xd(1,2,3);
    auto X = record_matrix(xd, t);
    std::vector<V1> F{X(0)+X(1), X(1)+X(2), X(2)+X(0)};
    auto J = sparse_jacobian<double>(F, X);
    Eigen::Matrix3d expected;
    expected << 1,1,0,
                0,1,1,
                1,0,1;
    CHECK((Eigen::MatrixXd(J) - expected).norm() < TOL);
}

TEST_CASE("sparse jacobian: correct shape m x n")
{
    Tape<double> t;
    Eigen::Vector4d xd(1,2,3,4);
    auto X = record_matrix(xd, t);
    std::vector<V1> F{X(0)+X(1), X(2)*X(3)};  // 2-output, 4-input
    auto J = sparse_jacobian<double>(F, X);
    CHECK(J.rows() == 2);
    CHECK(J.cols() == 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("constant Var: constant+constant doesn't grow tape")
{
    Tape<double> t;
    V1 x = var(2.0, t);
    size_t before = t.size();
    V1 c1(3.0), c2(4.0);
    V1 _ = c1 + c2;                 // both untracked: no node added
    CHECK(t.size() == before);
}

TEST_CASE("record_scalar: single tape → tracked Var<double>")
{
    Tape<double> t;
    V1 x = record_scalar(3.14, t);
    CHECK(x.getValue() == doctest::Approx(3.14));
    CHECK(t.size() == 1);
    CHECK(grad1(x, x) == doctest::Approx(1.0));
}

TEST_CASE("record_scalar: two tapes → Var<Var<double>> with nested tracking")
{
    Tape<double> t1; Tape<V1> t2;
    V2 x = record_scalar(3.14, t1, t2);
    CHECK(x.getValue().getValue() == doctest::Approx(3.14));
    CHECK(t1.size() == 1);
    CHECK(t2.size() == 1);
    // First derivative should equal 1
    auto outer = x.grad();
    CHECK(outer[x.getIndex()].getValue() == doctest::Approx(1.0));
}

TEST_CASE("iterateMatrix: gradient correct for row-major input")
{
    Tape<double> t;
    Eigen::Matrix<double,2,2,Eigen::RowMajor> Xd;
    Xd << 1,2,3,4;
    auto X = record_matrix(Xd, t);
    auto [dX] = gradient(X.sum(), X);
    CHECK((dX - Eigen::MatrixXd::Ones(2,2)).norm() < TOL);
}

TEST_CASE("iterateMatrix: hessian correct for row-major input")
{
    Tape<double> t1; Tape<V1> t2;
    Eigen::Matrix<double,2,2,Eigen::RowMajor> Xd;
    Xd << 1,2,3,4;
    auto X = record_matrix(Xd, t1, t2);
    auto H = hessian(X.squaredNorm(), X);
    CHECK((H - 2.0*Eigen::Matrix<double,4,4>::Identity()).norm() < TOL);
}
