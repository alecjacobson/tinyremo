// Probe: sparse_hessian step count for spring-energy variants with lambda.
// Investigates which positions of a shared lambda variable cause O(N^2) inner-tape
// work vs. O(N). Counts sparse_grad().size() summed over all input variables as a
// proxy for total work in sparse_hessian (same traversal, no allocator overhead).
//
// Build:
//   g++ -std=c++20 -O2 -I/path/to/eigen spring_lambda_probe.cpp -o spring_lambda_probe

#include "tinyremo_eigen.h"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>

using namespace tinyremo;
using V1 = Var<double>;
using V2 = Var<V1>;
using VecXd  = Eigen::VectorXd;
using VecV2  = Eigen::Matrix<V2, Eigen::Dynamic, 1>;

// ─────────────────────────────────────────────────────────────────────────────
// 1D circular spring energy variants (quadratic, no sqrt — cleaner t1 depth)
//   x has N entries, spring i connects x(i) and x((i+1)%N).
// ─────────────────────────────────────────────────────────────────────────────

// Baseline: no lambda.
template<typename T>
T spring_plain(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, int N)
{
    T E(0);
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        T d = x(j) - x(i);
        E = E + T(0.5)*d*d;
    }
    return E;
}

// lambda multiplies the whole sum from outside: f = lam * Σ E_i.
// lam enters the t2 backward once at the top node; each x(k) adjoint picks
// up one binary t1 node for the lam factor → O(1) per variable → O(N) total.
template<typename T>
T spring_outer_lambda(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, T lam, int N)
{
    return lam * spring_plain(x, N);
}

// lambda multiplies each term inside the sum: f = Σ lam*E_i.
// Algebraically identical to spring_outer_lambda (same Hessian, same graph),
// so same O(N) performance.
template<typename T>
T spring_inner_lambda(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, T lam, int N)
{
    T E(0);
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        T d = x(j) - x(i);
        E = E + lam * (T(0.5)*d*d);
    }
    return E;
}

// lambda appears inside each spring as a coefficient on one endpoint:
//   f = Σ 0.5*(x(j) - lam*x(i))^2
// Each x(k) adjoint has O(1) lam-related nodes on t1; lam's adjoint accumulates
// O(N) terms but in a single sparse_grad call → O(N) total.
template<typename T>
T spring_lambda_factor(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, T lam, int N)
{
    T E(0);
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        T d = x(j) - lam*x(i);
        E = E + T(0.5)*d*d;
    }
    return E;
}

// lambda enters via a recurrence: w_0=lam, w_i = w_{i-1}*lam (i.e. w_i = lam^(i+1)).
// f = Σ w_i * E_i.
// After i multiplications, w_i.value on t1 is a chain of i binary nodes.
// So adj[x(k).t2] = w_k.value * dE_k/dx(k) has O(k) t1 depth.
// Total sparse_grad steps: Σ O(k) = O(N^2).
template<typename T>
T spring_lambda_accumulated(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, T lam, int N)
{
    T E(0);
    T w = lam;
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        T d = x(j) - x(i);
        E = E + w * (T(0.5)*d*d);
        w = w * lam;
    }
    return E;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step counter: sum sparse_grad().size() over x(0..N-1) and lam.
// ─────────────────────────────────────────────────────────────────────────────

static size_t count_steps(V2 f, const VecV2& x, V2 lam)
{
    auto df_d = f.grad();
    size_t total = 0;
    for (int k = 0; k < (int)x.size(); k++)
        total += df_d[x(k).getIndex()].sparse_grad().size();
    total += df_d[lam.getIndex()].sparse_grad().size();
    return total;
}

// No-lambda variant: count only x variables.
static size_t count_steps_no_lam(V2 f, const VecV2& x)
{
    auto df_d = f.grad();
    size_t total = 0;
    for (int k = 0; k < (int)x.size(); k++)
        total += df_d[x(k).getIndex()].sparse_grad().size();
    return total;
}

static VecXd make_x(int N)
{
    VecXd xd(N);
    for (int i = 0; i < N; i++) xd(i) = 1.0 + 0.1*std::sin(i+1.0);
    return xd;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main: print step counts and doubling ratios
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    const int Ns[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    const double lam_val = 1.5;

    // Column header
    printf("%-30s", "variant");
    for (int N : Ns) printf(" %8d", N);
    printf("  scaling\n");

    // ── Helper to print one row ───────────────────────────────────────────────
    auto print_row = [&](const char* name, auto fn) {
        printf("%-30s", name);
        size_t prev = 0;
        for (int N : Ns) {
            VecXd xd = make_x(N);
            size_t steps = fn(N, xd);
            printf(" %8zu", steps);
            prev = steps;
        }
        // Doubling ratio between last two Ns
        {
            constexpr int kNs = sizeof(Ns)/sizeof(Ns[0]);
        int N1 = Ns[kNs-2], N2 = Ns[kNs-1];
            VecXd xd1 = make_x(N1), xd2 = make_x(N2);
            size_t s1 = fn(N1, xd1), s2 = fn(N2, xd2);
            printf("  ratio=%.2f (%s)\n", (double)s2/(double)s1,
                   s2 < 2.5*s1 ? "O(N)" : s2 < 3.5*s1 ? "O(N^2?)" : "O(N^2)");
        }
    };

    // 1. Baseline — no lambda
    print_row("spring_plain", [&](int N, const VecXd& xd) -> size_t {
        Tape<double> t1; Tape<V1> t2;
        auto x = record_matrix(xd, t1, t2);
        return count_steps_no_lam(spring_plain(x, N), x);
    });

    // 2. Lambda outside the sum
    print_row("spring_outer_lambda", [&](int N, const VecXd& xd) -> size_t {
        Tape<double> t1; Tape<V1> t2;
        auto x   = record_matrix(xd, t1, t2);
        V2   lam = record_scalar(lam_val, t1, t2);
        return count_steps(spring_outer_lambda(x, lam, N), x, lam);
    });

    // 3. Lambda inside each term (algebraically identical to outer)
    print_row("spring_inner_lambda", [&](int N, const VecXd& xd) -> size_t {
        Tape<double> t1; Tape<V1> t2;
        auto x   = record_matrix(xd, t1, t2);
        V2   lam = record_scalar(lam_val, t1, t2);
        return count_steps(spring_inner_lambda(x, lam, N), x, lam);
    });

    // 4. Lambda as coefficient inside each spring (lam*x_i)
    print_row("spring_lambda_factor", [&](int N, const VecXd& xd) -> size_t {
        Tape<double> t1; Tape<V1> t2;
        auto x   = record_matrix(xd, t1, t2);
        V2   lam = record_scalar(lam_val, t1, t2);
        return count_steps(spring_lambda_factor(x, lam, N), x, lam);
    });

    // 5. Lambda via recurrence: w_i = lam^(i+1) → O(N^2)
    print_row("spring_lambda_accumulated", [&](int N, const VecXd& xd) -> size_t {
        Tape<double> t1; Tape<V1> t2;
        auto x   = record_matrix(xd, t1, t2);
        V2   lam = record_scalar(lam_val, t1, t2);
        return count_steps(spring_lambda_accumulated(x, lam, N), x, lam);
    });

    return 0;
}
