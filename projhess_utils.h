#pragma once
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include "tinyremo_eigen.h"
#include <Eigen/Dense>

// Project a symmetric matrix to PSD by clamping negative eigenvalues to eps.
inline Eigen::MatrixXd project_psd(const Eigen::MatrixXd& H, double eps = 1e-7)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
    return eig.eigenvectors()
         * eig.eigenvalues().cwiseMax(eps).asDiagonal()
         * eig.eigenvectors().transpose();
}

// Build a Var<Var<double>> on the MAIN tapes representing a quadratic that
// matches f at xi_d to second order with the Hessian projected to PSD.
//
// Uses a single nested-tape evaluation to get primal, gradient and Hessian
// together, avoiding two separate calls to f.
template <typename XiMain, typename Func>
tinyremo::Var<tinyremo::Var<double>>
synthetic_quadratic(const XiMain& xi_main, Func&& f,
                    const Eigen::VectorXd& xi_d, double eps = 1e-7)
{
    using V1 = tinyremo::Var<double>;
    using V2 = tinyremo::Var<V1>;

    const int k = (int)xi_d.size();

    // One nested-tape pass: get primal, gradient and Hessian together.
    tinyremo::Tape<double> t1; tinyremo::Tape<V1> t2;
    auto xi_f  = tinyremo::record_matrix(xi_d, t1, t2);
    V2   yi_f  = f(xi_f);

    double primal = yi_f.getValue().getValue();

    // Outer backward (tape2): grows t1 with derivative-tracking nodes.
    // Zero-check is intentionally disabled for Var<double> scalar type —
    // see grad() implementation: is_arithmetic_v<Var<double>> is false.
    auto df_d_raw = yi_f.grad();   // vector<V1>, indexed by tape2 position

    // Gradient: primal value of each outer-gradient entry
    Eigen::VectorXd grad_d(k);
    for (int j = 0; j < k; j++)
        grad_d(j) = df_d_raw[xi_f(j).getIndex()].getValue();

    // Hessian: inner backward on (grown) t1 for each input direction
    Eigen::MatrixXd H(k, k);
    std::vector<double> inner_buf;
    for (int j = 0; j < k; j++) {
        df_d_raw[xi_f(j).getIndex()].grad(inner_buf);
        for (int l = 0; l < k; l++)
            H(j, l) = inner_buf[xi_f(l).getValue().getIndex()];
    }

    // Project
    Eigen::MatrixXd pH = project_psd(H, eps);

    // Build synthetic quadratic on the main tapes:
    // g(xi) = cnst + lin^T xi + 0.5 xi^T pH xi
    // so that g(xi_d)=primal, ∇g(xi_d)=grad_d, ∂²g/∂xi²=pH everywhere.
    Eigen::VectorXd lin  = grad_d - pH * xi_d;
    double          cnst = primal - grad_d.dot(xi_d) + 0.5 * xi_d.dot(pH * xi_d);

    V2 result(cnst);
    for (int j = 0; j < k; j++)
        result += V2(lin[j]) * xi_main(j);
    for (int j = 0; j < k; j++)
        for (int l = j; l < k; l++) {
            double c = pH(j, l) * (j == l ? 0.5 : 1.0);
            if (std::abs(c) > 1e-15)
                result += V2(c) * xi_main(j) * xi_main(l);
        }
    return result;
}
