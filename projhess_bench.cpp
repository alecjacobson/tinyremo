#include "projhess_utils.h"
#include <Eigen/Sparse>
#include <chrono>
#include <cstdio>
#include <functional>
#include <limits>
#include <numbers>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Material constants
// ─────────────────────────────────────────────────────────────────────────────
static const double MU     = 1.0;
static const double LAMBDA = 4.0;

// ─────────────────────────────────────────────────────────────────────────────
// Grid mesh: n×n vertices, 2(n-1)² triangles
// ─────────────────────────────────────────────────────────────────────────────
static void make_grid(int n, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    V.resize(n * n, 2);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V.row(i * n + j) << (double)j, (double)i;

    F.resize(2 * (n-1) * (n-1), 3);
    int fi = 0;
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-1; j++) {
            int v00 = i*n+j, v10 = i*n+j+1, v01 = (i+1)*n+j, v11 = (i+1)*n+j+1;
            F.row(fi++) << v00, v10, v11;
            F.row(fi++) << v00, v11, v01;
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-element precomputed data
// ─────────────────────────────────────────────────────────────────────────────
struct ElemData { Eigen::Matrix2d Mr_inv; double A; };

static std::vector<ElemData> precompute(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    std::vector<ElemData> data(F.rows());
    for (int f = 0; f < F.rows(); f++) {
        Eigen::Vector2d a = V.row(F(f,0)), b = V.row(F(f,1)), c = V.row(F(f,2));
        Eigen::Matrix2d Mr;
        Mr.col(0) = b - a;
        Mr.col(1) = c - a;
        data[f].A       = 0.5 * std::abs(Mr.determinant());
        data[f].Mr_inv  = Mr.inverse();
    }
    return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D stable Neo-Hookean element energy
//
//   W = μ/2 (Ic - 2) + λ/2 (detF - α)²    α = 1 + μ/λ
//
// Adapted from the 3D tet version in the TinyAD example by replacing
// 3×3 matrices/vectors with 2×2/2D and Ic-3 → Ic-2 (2D identity trace).
// Each triangle provides local 6-vector [a.x, a.y, b.x, b.y, c.x, c.y].
// ─────────────────────────────────────────────────────────────────────────────
template <typename Derived>
typename Derived::Scalar
neohookean_2d(const Eigen::MatrixBase<Derived>& xl,
              const Eigen::Matrix2d& Mr_inv, double A)
{
    using T = typename Derived::Scalar;

    Eigen::Matrix<T, 2, 1> a = xl.template segment<2>(0);
    Eigen::Matrix<T, 2, 1> b = xl.template segment<2>(2);
    Eigen::Matrix<T, 2, 1> c = xl.template segment<2>(4);

    Eigen::Matrix<T, 2, 2> M;
    M.col(0) = b - a;
    M.col(1) = c - a;

    // Deformation gradient J = M * Mr⁻¹  (precomputed, cast to T for AD)
    Eigen::Matrix<T, 2, 2> J = M * Mr_inv.template cast<T>();

    T Ic   = (J.transpose() * J).trace();
    T detF = J.determinant();
    double alpha = 1.0 + MU / LAMBDA;
    T W = T(MU/2.0) * (Ic - T(2.0))
        + T(LAMBDA/2.0) * (detF - T(alpha)) * (detF - T(alpha));
    return T(A) * W;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build flat position vector from vertex matrix
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::VectorXd flatten(const Eigen::MatrixXd& V)
{
    const int nV = V.rows();
    Eigen::VectorXd x(2 * nV);
    for (int i = 0; i < nV; i++) { x(2*i) = V(i,0); x(2*i+1) = V(i,1); }
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// Trivial projected Hessian: per-element fresh tapes, project, scatter
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::SparseMatrix<double>
trivial_proj_hessian(const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     const std::vector<ElemData>& elems,
                     double eps = 1e-7)
{
    using V1 = tinyremo::Var<double>;

    const int nV = V.rows();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(F.rows() * 36);

    for (int f = 0; f < F.rows(); f++) {
        Eigen::VectorXd xl(6);
        for (int i = 0; i < 3; i++) {
            xl(2*i)   = V(F(f,i), 0);
            xl(2*i+1) = V(F(f,i), 1);
        }

        tinyremo::Tape<double> t1; tinyremo::Tape<V1> t2;
        auto xl_ad = tinyremo::record_matrix(xl, t1, t2);
        auto Hi    = tinyremo::hessian(
                         neohookean_2d(xl_ad, elems[f].Mr_inv, elems[f].A),
                         xl_ad);
        auto pHi   = project_psd(Hi, eps);

        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
                for (int da = 0; da < 2; da++)
                    for (int db = 0; db < 2; db++)
                        triplets.emplace_back(
                            2*F(f,a)+da, 2*F(f,b)+db,
                            pHi(2*a+da, 2*b+db));
    }

    Eigen::SparseMatrix<double> H(2*nV, 2*nV);
    H.setFromTriplets(triplets.begin(), triplets.end());
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// synthetic_quadratic projected Hessian: build main tape, call sparse_hessian
// ─────────────────────────────────────────────────────────────────────────────
static Eigen::SparseMatrix<double>
synthetic_proj_hessian(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXi& F,
                       const std::vector<ElemData>& elems,
                       double eps = 1e-7)
{
    using V1 = tinyremo::Var<double>;
    using V2 = tinyremo::Var<V1>;

    const int nV   = V.rows();
    Eigen::VectorXd x_d = flatten(V);

    tinyremo::Tape<double> t1_main; tinyremo::Tape<V1> t2_main;
    auto x = tinyremo::record_matrix(x_d, t1_main, t2_main);

    V2 f(0);
    for (int fi = 0; fi < F.rows(); fi++) {
        Eigen::VectorXd xi_d(6);
        Eigen::Matrix<V2, Eigen::Dynamic, 1> xi_main(6);
        for (int i = 0; i < 3; i++) {
            int vi = F(fi, i);
            xi_d(2*i)     = V(vi, 0);  xi_d(2*i+1)    = V(vi, 1);
            xi_main(2*i)  = x(2*vi);   xi_main(2*i+1) = x(2*vi+1);
        }
        const ElemData& ed = elems[fi];
        f += synthetic_quadratic(xi_main,
                 [&ed](auto& v){ return neohookean_2d(v, ed.Mr_inv, ed.A); },
                 xi_d, eps);
    }

    return tinyremo::sparse_hessian(f, x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────────────────────────────────────
static double now_s()
{
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static volatile double g_sink = 0;

// Run f() adaptively to fill ~0.5 s; return {seconds/call, iters}.
static std::pair<double,int> bench(std::function<void()> f)
{
    f();                                    // warm-up
    double t0 = now_s(); f(); double t1 = now_s();
    int iters = std::clamp((int)(0.5 / std::max(t1-t0, 1e-9)), 2, 500);
    t0 = now_s();
    for (int i = 0; i < iters; i++) f();
    return {(now_s()-t0)/iters, iters};
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("%6s %9s %7s %14s %14s\n",
           "# nV", "nF", "iters", "trivial(s)", "synthetic(s)");

    for (int n = 2; n <= 64; n *= 2) {
        Eigen::MatrixXd V0; Eigen::MatrixXi F;
        make_grid(n, V0, F);

        // Perturb vertices slightly so energy/Hessian is non-trivial
        Eigen::MatrixXd V = V0;
        for (int i = 0; i < V.rows(); i++) {
            V(i,0) += 0.05 * std::sin(3.0*i + 1.0);
            V(i,1) += 0.05 * std::cos(5.0*i + 2.0);
        }

        auto elems = precompute(V0, F);   // rest shapes from unperturbed grid

        auto [t_tri,  iters] = bench([&]{
            auto H = trivial_proj_hessian(V, F, elems);
            g_sink += H.coeff(0,0);
        });
        auto [t_syn, _] = bench([&]{
            auto H = synthetic_proj_hessian(V, F, elems);
            g_sink += H.coeff(0,0);
        });

        printf("%6d %9d %7d %14.4e %14.4e\n",
               (int)V.rows(), (int)F.rows(), iters, t_tri, t_syn);
    }
}
