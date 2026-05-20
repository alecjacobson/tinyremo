#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "projhess_utils.h"   // project_psd, synthetic_quadratic
#include <Eigen/Sparse>

// ─────────────────────────────────────────────────────────────────────────────
// Toy element function: f(a, b) = (ab - 1)²
//
// Exact element Hessian:  [[2b², 4ab-2], [4ab-2, 2a²]]
// At most points this is INDEFINITE (det = 4b²a² - (4ab-2)² < 0),
// so PSD projection genuinely changes the matrix.
// ─────────────────────────────────────────────────────────────────────────────

template <typename Derived>
typename Derived::Scalar elem_func(const Eigen::MatrixBase<Derived>& xi)
{
    using T = typename Derived::Scalar;
    T d = xi(0) * xi(1) - T(1);
    return d * d;
}

// Analytical element Hessian (double, for reference checks)
Eigen::Matrix2d elem_hessian_analytical(double a, double b)
{
    Eigen::Matrix2d H;
    H << 2*b*b,       4*a*b - 2,
         4*a*b - 2,   2*a*a;
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// Assembly helpers used by both approaches
// ─────────────────────────────────────────────────────────────────────────────

// Trivial per-element projected Hessian: fresh nested tapes per element.
Eigen::MatrixXd projected_hessian_trivial(
    const Eigen::VectorXd& x_d,
    const std::vector<std::vector<int>>& elements,
    double eps = 1e-7)
{
    const int n = (int)x_d.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);

    for (const auto& elem : elements) {
        const int k = (int)elem.size();
        Eigen::VectorXd xi_d(k);
        for (int i = 0; i < k; i++) xi_d(i) = x_d(elem[i]);

        using V1 = tinyremo::Var<double>;
        tinyremo::Tape<double> t1; tinyremo::Tape<V1> t2;
        auto xi  = tinyremo::record_matrix(xi_d, t1, t2);
        auto Hi  = tinyremo::hessian(elem_func(xi), xi);
        auto pHi = project_psd(Hi, eps);

        for (int a = 0; a < k; a++)
            for (int b = 0; b < k; b++)
                H(elem[a], elem[b]) += pHi(a, b);
    }
    return H;
}

// synthetic_quadratic-based projected Hessian.
Eigen::MatrixXd projected_hessian_synthetic(
    const Eigen::VectorXd& x_d,
    const std::vector<std::vector<int>>& elements,
    double eps = 1e-7)
{
    using V1 = tinyremo::Var<double>;
    using V2 = tinyremo::Var<V1>;

    tinyremo::Tape<double> t1_main; tinyremo::Tape<V1> t2_main;
    auto x = tinyremo::record_matrix(x_d, t1_main, t2_main);

    V2 f(0);
    for (const auto& elem : elements) {
        const int k = (int)elem.size();
        Eigen::VectorXd xi_d_local(k);
        Eigen::Matrix<V2, Eigen::Dynamic, 1> xi_main(k);
        for (int i = 0; i < k; i++) {
            xi_d_local(i) = x_d(elem[i]);
            xi_main(i)    = x(elem[i]);
        }
        f += synthetic_quadratic(xi_main,
                 [](auto& v){ return elem_func(v); },
                 xi_d_local, eps);
    }
    return tinyremo::hessian(f, x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("element Hessian: tinyremo matches analytical")
{
    // Verify the element Hessian of elem_func against the known formula.
    double a = 1.0, b = 2.0;
    Eigen::Vector2d xi_d(a, b);

    using V1 = tinyremo::Var<double>;
    tinyremo::Tape<double> t1; tinyremo::Tape<V1> t2;
    auto xi = tinyremo::record_matrix(xi_d, t1, t2);
    auto H  = tinyremo::hessian(elem_func(xi), xi);

    auto H_exact = elem_hessian_analytical(a, b);
    CHECK((H - H_exact).norm() < 1e-10);
}

TEST_CASE("element Hessian: indefinite at test point (projection is non-trivial)")
{
    // Confirm the unmodified element Hessian is NOT PSD at (1,2) and (3,4),
    // so that the projected Hessian tests are actually exercising the projection.
    for (auto [a, b] : std::initializer_list<std::pair<double,double>>{{1,2},{3,4}}) {
        auto H = elem_hessian_analytical(a, b);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(H);
        CHECK(eig.eigenvalues().minCoeff() < 0);  // genuinely indefinite
    }
}

TEST_CASE("projected hessian: trivial matches synthetic (non-overlapping elements)")
{
    // 4 DOFs, 2 non-overlapping elements: {0,1} and {2,3}
    Eigen::Vector4d x_d(1, 2, 3, 4);
    std::vector<std::vector<int>> elements = {{0,1}, {2,3}};

    auto H_trivial   = projected_hessian_trivial(x_d, elements);
    auto H_synthetic = projected_hessian_synthetic(x_d, elements);

    CHECK((H_trivial - H_synthetic).norm() < 1e-10);
}

TEST_CASE("projected hessian: trivial matches synthetic (overlapping chain)")
{
    // 5 DOFs, 4 overlapping pair elements along a chain: {0,1},{1,2},{2,3},{3,4}
    Eigen::VectorXd x_d(5); x_d << 1, 2, 3, 4, 5;
    std::vector<std::vector<int>> elements = {{0,1},{1,2},{2,3},{3,4}};

    auto H_trivial   = projected_hessian_trivial(x_d, elements);
    auto H_synthetic = projected_hessian_synthetic(x_d, elements);

    CHECK((H_trivial - H_synthetic).norm() < 1e-10);
}

TEST_CASE("projected hessian: result is PSD")
{
    Eigen::VectorXd x_d(5); x_d << 1, 2, 3, 4, 5;
    std::vector<std::vector<int>> elements = {{0,1},{1,2},{2,3},{3,4}};

    auto H = projected_hessian_trivial(x_d, elements);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
    CHECK(eig.eigenvalues().minCoeff() >= -1e-9);
}

TEST_CASE("projected hessian: gradient from synthetic matches true gradient")
{
    // The synthetic tape should give exactly ∇f(x_d), not ∇g(x_d) from some
    // approximate function.
    Eigen::VectorXd x_d(5); x_d << 1, 2, 3, 4, 5;
    std::vector<std::vector<int>> elements = {{0,1},{1,2},{2,3},{3,4}};

    // True gradient via a plain first-order tape over the original f
    Eigen::VectorXd true_grad;
    {
        tinyremo::Tape<double> t;
        auto x = tinyremo::record_matrix(x_d, t);
        tinyremo::Var<double> f(0);
        for (const auto& elem : elements) {
            const int k = (int)elem.size();
            Eigen::Matrix<tinyremo::Var<double>, Eigen::Dynamic, 1> xi(k);
            for (int i = 0; i < k; i++) xi(i) = x(elem[i]);
            f += elem_func(xi);
        }
        auto [gv] = tinyremo::gradient(f, x);
        true_grad = gv;
    }

    // Gradient extracted from the synthetic nested tape
    Eigen::VectorXd synth_grad;
    {
        using V1 = tinyremo::Var<double>;
        using V2 = tinyremo::Var<V1>;

        tinyremo::Tape<double> t1_main; tinyremo::Tape<V1> t2_main;
        auto x = tinyremo::record_matrix(x_d, t1_main, t2_main);

        V2 f(0);
        for (const auto& elem : elements) {
            const int k = (int)elem.size();
            Eigen::VectorXd xi_d_local(k);
            Eigen::Matrix<V2, Eigen::Dynamic, 1> xi_main(k);
            for (int i = 0; i < k; i++) {
                xi_d_local(i) = x_d(elem[i]);
                xi_main(i)    = x(elem[i]);
            }
            f += synthetic_quadratic(xi_main,
                     [](auto& v){ return elem_func(v); },
                     xi_d_local);
        }
        // gradient() on f : Var<Var<double>> returns Matrix<Var<double>,...>
        auto [gv] = tinyremo::gradient(f, x);
        synth_grad = gv.unaryExpr(
            [](const V1& v){ return v.getValue(); });
    }

    CHECK((synth_grad - true_grad).norm() < 1e-10);
}

TEST_CASE("projected hessian: primal value from synthetic matches true primal")
{
    Eigen::VectorXd x_d(5); x_d << 1, 2, 3, 4, 5;
    std::vector<std::vector<int>> elements = {{0,1},{1,2},{2,3},{3,4}};

    // True primal
    double true_f = 0;
    for (const auto& elem : elements) {
        Eigen::VectorXd xi_d(elem.size());
        for (int i = 0; i < (int)elem.size(); i++) xi_d(i) = x_d(elem[i]);
        true_f += elem_func(xi_d);
    }

    // Primal from synthetic tape
    {
        using V1 = tinyremo::Var<double>;
        using V2 = tinyremo::Var<V1>;

        tinyremo::Tape<double> t1_main; tinyremo::Tape<V1> t2_main;
        auto x = tinyremo::record_matrix(x_d, t1_main, t2_main);

        V2 f(0);
        for (const auto& elem : elements) {
            const int k = (int)elem.size();
            Eigen::VectorXd xi_d_local(k);
            Eigen::Matrix<V2, Eigen::Dynamic, 1> xi_main(k);
            for (int i = 0; i < k; i++) {
                xi_d_local(i) = x_d(elem[i]);
                xi_main(i)    = x(elem[i]);
            }
            f += synthetic_quadratic(xi_main,
                     [](auto& v){ return elem_func(v); },
                     xi_d_local);
        }
        double synth_f = f.getValue().getValue();
        CHECK(synth_f == doctest::Approx(true_f).epsilon(1e-12));
    }
}
