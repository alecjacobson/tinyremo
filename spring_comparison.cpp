#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include "tinyremo_eigen.h"
#include <TinyAD/ScalarFunction.hh>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <numbers>
#include <limits>
#include <cstdio>

static double tictoc()
{
    double t = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    static double t0 = t;
    double dt = t - t0;
    t0 = t;
    return dt;
}

// Regular N-gon with unit edge lengths
static Eigen::VectorXd init_x(int N)
{
    const double r = 1.0 / (2.0 * std::sin(std::numbers::pi / N));
    Eigen::VectorXd x(2 * N);
    for (int i = 0; i < N; i++)
    {
        x(2*i)   = r * std::cos(2.0 * std::numbers::pi * i / N);
        x(2*i+1) = r * std::sin(2.0 * std::numbers::pi * i / N);
    }
    return x;
}

// Total energy of a loop of N unit springs over a flat 2N position vector
template <typename T>
T spring_energy(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, int N)
{
    T E(0);
    for (int i = 0; i < N; i++)
    {
        int j = (i + 1) % N;
        T dx = x(2*j)   - x(2*i);
        T dy = x(2*j+1) - x(2*i+1);
        T d  = sqrt(dx*dx + dy*dy) - T(1);
        E += T(0.5) * d * d;
    }
    return E;
}

// Finite differences (centered, eps=1e-5)
Eigen::MatrixXd fd_hessian(const Eigen::VectorXd& x, int N)
{
    const int n = 2 * N;
    const double eps = 1e-5;
    auto grad = [&](const Eigen::VectorXd& xv)
    {
        Eigen::VectorXd g(n);
        for (int i = 0; i < n; i++)
        {
            Eigen::VectorXd xp = xv, xm = xv;
            xp(i) += eps; xm(i) -= eps;
            g(i) = (spring_energy(xp, N) - spring_energy(xm, N)) / (2*eps);
        }
        return g;
    };
    Eigen::MatrixXd H(n, n);
    for (int i = 0; i < n; i++)
    {
        Eigen::VectorXd xp = x, xm = x;
        xp(i) += eps; xm(i) -= eps;
        H.col(i) = (grad(xp) - grad(xm)) / (2*eps);
    }
    return H;
}

// TinyAD: scatter-gather with ScalarFunction, returns sparse Hessian
Eigen::SparseMatrix<double> tinyad_hessian(const Eigen::VectorXd& x_d, int N)
{
    auto func = TinyAD::scalar_function<2>(TinyAD::range(N));
    func.add_elements<2>(TinyAD::range(N), [&](auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        int i = element.handle;
        int j = (i + 1) % N;
        Eigen::Vector2<T> pi = element.variables(i);
        Eigen::Vector2<T> pj = element.variables(j);
        T d = (pj - pi).norm() - T(1);
        return T(0.5) * d * d;
    });
    Eigen::VectorXd x = func.x_from_data([&](int i) -> Eigen::Vector2d
    {
        return x_d.segment<2>(2 * i);
    });
    return func.eval_hessian(x);
}

// tinyremo: dense Hessian, global reverse-mode over all 2N DOFs
Eigen::MatrixXd tinyremo_hessian(const Eigen::VectorXd& x_d, int N)
{
    tinyremo::Tape<double> tape1;
    tinyremo::Tape<tinyremo::Var<double>> tape2;
    auto x = tinyremo::record_matrix(x_d, tape1, tape2);
    auto E = spring_energy(x, N);
    return tinyremo::hessian(E, x);
}

// tinyremo: sparse Hessian, global reverse-mode over all 2N DOFs
Eigen::SparseMatrix<double> tinyremo_sparse_hessian(const Eigen::VectorXd& x_d, int N)
{
    tinyremo::Tape<double> tape1;
    tinyremo::Tape<tinyremo::Var<double>> tape2;
    auto x = tinyremo::record_matrix(x_d, tape1, tape2);
    auto E = spring_energy(x, N);
    return tinyremo::sparse_hessian(E, x);
}

int main()
{
    // Correctness check
    {
        const int N = 6;
        auto x = init_x(N);
        auto H_fd       = fd_hessian(x, N);
        auto H_tinyad          = tinyad_hessian(x, N);
        auto H_tinyremo        = tinyremo_hessian(x, N);
        auto H_tinyremo_sparse = tinyremo_sparse_hessian(x, N);
        printf("# N=%d  max|tinyad-fd|=%g  max|tinyremo-fd|=%g  max|tinyremo_sparse-fd|=%g\n", N,
            (Eigen::MatrixXd(H_tinyad)          - H_fd).cwiseAbs().maxCoeff(),
            (H_tinyremo                         - H_fd).cwiseAbs().maxCoeff(),
            (Eigen::MatrixXd(H_tinyremo_sparse) - H_fd).cwiseAbs().maxCoeff());
    }

    // Benchmark (TinyAD and tinyremo; FD is O(N^3) so omitted)
    printf("# 2D unit-spring loop Hessian timings (seconds/call)\n");
    printf("# N tinyad tinyremo_dense tinyremo_sparse\n");
    bool skip_dense = false;
    for (int N = 4; N <= 65536; N *= 2)
    {
        auto x = init_x(N);
        const int iters = std::clamp(6400 / N, 2, 100);

        tictoc();
        for (int it = 0; it < iters; it++) tinyad_hessian(x, N);
        double t_tinyad = tictoc() / iters;

        double t_dense = std::numeric_limits<double>::quiet_NaN();
        if (!skip_dense)
        {
            tictoc();
            for (int it = 0; it < iters; it++) tinyremo_hessian(x, N);
            t_dense = tictoc() / iters;
            if (t_dense > 1.0) skip_dense = true;
        }

        tictoc();
        for (int it = 0; it < iters; it++) tinyremo_sparse_hessian(x, N);
        double t_sparse = tictoc() / iters;

        printf("%d %g %g %g\n", N, t_tinyad, t_dense, t_sparse);
    }
}
