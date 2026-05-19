#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include "tinyremo_eigen.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <numbers>
#include <limits>
#include <cstdio>
#include <functional>
#include <set>

using V1 = tinyremo::Var<double>;
using V2 = tinyremo::Var<V1>;
using namespace tinyremo;

// ─────────────────────────────────────────────────────────────────────────────
// Timing — adaptive: measure one call, then run enough to fill ~0.5 s
// ─────────────────────────────────────────────────────────────────────────────

static volatile double g_sink = 0;

static double now()
{
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// Returns {seconds_per_call, iters_used}.
static std::pair<double,int> run(std::function<void()> f)
{
    f();                                 // one warm-up
    double t0 = now(); f(); double t1 = now();
    double t_one = t1 - t0;
    int iters = std::clamp((int)(0.5 / std::max(t_one, 1e-9)), 2, 500);
    t0 = now();
    for (int i = 0; i < iters; i++) f();
    return { (now()-t0)/iters, iters };
}

// ─────────────────────────────────────────────────────────────────────────────
// Functions under test
// ─────────────────────────────────────────────────────────────────────────────

// Loop of N 2D unit springs — exercises operator-, sqrt.  Sparse Hessian.
template <typename T>
T spring_energy(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, int N)
{
    T E(0);
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        T dx = x(2*j)-x(2*i), dy = x(2*j+1)-x(2*i+1);
        T d  = sqrt(dx*dx + dy*dy) - T(1);
        E   += T(0.5)*d*d;
    }
    return E;
}

// Differentiate through Cholesky — exercises operator/, complex graph. Dense Hessian.
template <typename T>
T llt_func(const Eigen::Matrix<T,Eigen::Dynamic,1>& x)
{
    int N = (int)x.size();
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> A(N,N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) A(i,j) = x(i)+x(j);
        A(i,i) = T(2)*A(i,i);
    }
    return A.llt().solve(x).eval().squaredNorm();
}

// Generalised Rosenbrock — exercises operator- heavily.  Tridiagonal Hessian.
template <typename T>
T rosenbrock(const Eigen::Matrix<T,Eigen::Dynamic,1>& x)
{
    T E(0);
    for (int i = 0; i < (int)x.size()-1; i++) {
        T a = x(i+1)-x(i)*x(i), b = T(1)-x(i);
        E += T(100)*a*a + b*b;
    }
    return E;
}

// Circular sin/cos/exp — exercises transcendental double-evaluation.  Sparse Hessian.
template <typename T>
T trig_func(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, int N)
{
    T E(0);
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        E += sin(x(i))*cos(x(j)) + exp(x(i)-x(j));
    }
    return E;
}

// Circular stencil: F[i] = sin(x[i])*cos(x[(i+1)%N]) + x[i]^2.
// Each output touches exactly 2 inputs → bidiagonal Jacobian (2 nonzeros/row).
// This is the canonical use case for sparse_jacobian and sparse_grad.
template <typename T>
std::vector<T> stencil_func(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, int N)
{
    std::vector<T> F(N);
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        F[i] = sin(x(i))*cos(x(j)) + x(i)*x(i);
    }
    return F;
}

// Circular ratio (x_i-x_j)/(x_i+x_j) — exercises operator- and operator/.  Sparse Hessian.
template <typename T>
T ratio_func(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, int N)
{
    T E(0);
    for (int i = 0; i < N; i++) {
        int j = (i+1)%N;
        E += (x(i)-x(j))/(x(i)+x(j));
    }
    return E;
}

// Random graph: N 2D points, each connected to ~k random neighbours.
// Hessian is sparse but irregular (not banded).
// Uses a simple deterministic LCG for reproducibility.
struct RandomGraph {
    int N;
    std::vector<std::pair<int,int>> edges;  // undirected, (i<j)
    Eigen::VectorXd x;                      // 2N flat position vector
};

static RandomGraph make_random_graph(int N, int k_per_node = 4, uint64_t seed = 42)
{
    // Deterministic LCG (Knuth)
    auto rnd = [](uint64_t& s) -> uint64_t {
        return s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    };
    auto rnd_int = [&](uint64_t& s, int n) -> int {
        return (int)((rnd(s) >> 33) % (uint64_t)n);
    };
    auto rnd_double = [&](uint64_t& s) -> double {
        return (double)(rnd(s) >> 11) * (1.0 / (double)(1ULL << 53));
    };

    uint64_t s = seed;

    // Positions: uniform on [0, sqrt(N)] x [0, sqrt(N)]
    double scale = std::sqrt((double)N);
    Eigen::VectorXd x(2*N);
    for (int i = 0; i < 2*N; i++) x(i) = rnd_double(s) * scale;

    // Edges: for each node try to attach k_per_node unique neighbours
    std::set<std::pair<int,int>> seen;
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < N; i++) {
        for (int t = 0, added = 0; t < k_per_node*6 && added < k_per_node; ++t) {
            int j = rnd_int(s, N);
            if (j == i) continue;
            auto e = std::make_pair(std::min(i,j), std::max(i,j));
            if (seen.insert(e).second) { edges.push_back(e); ++added; }
        }
    }
    return {N, std::move(edges), std::move(x)};
}

template <typename T>
T random_spring_energy(const Eigen::Matrix<T,Eigen::Dynamic,1>& x,
                       const std::vector<std::pair<int,int>>& edges)
{
    T E(0);
    for (auto& [i,j] : edges) {
        T dx = x(2*j)-x(2*i), dy = x(2*j+1)-x(2*i+1);
        T d  = sqrt(dx*dx + dy*dy) - T(1);
        E   += T(0.5)*d*d;
    }
    return E;
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisers (deterministic)
// ─────────────────────────────────────────────────────────────────────────────

static Eigen::VectorXd spring_x(int N) {
    const double r = 1.0/(2.0*std::sin(std::numbers::pi/N));
    Eigen::VectorXd x(2*N);
    for (int i = 0; i < N; i++) {
        x(2*i)   = r*std::cos(2.0*std::numbers::pi*i/N) + 0.07*std::sin(i+1.0);
        x(2*i+1) = r*std::sin(2.0*std::numbers::pi*i/N) + 0.07*std::cos(i+2.0);
    }
    return x;
}
static Eigen::VectorXd llt_x(int N)   { Eigen::VectorXd x(N);   for(int i=0;i<N;i++) x(i)=i+1.0;                    return x; }
static Eigen::VectorXd rosen_x(int N) { Eigen::VectorXd x(N);   for(int i=0;i<N;i++) x(i)=1.0+0.1*std::sin(i+1.0); return x; }
static Eigen::VectorXd trig_x(int N)  { Eigen::VectorXd x(N);   for(int i=0;i<N;i++) x(i)=0.5+0.3*std::sin(i+1.0); return x; }
static Eigen::VectorXd ratio_x(int N) { Eigen::VectorXd x(N);   for(int i=0;i<N;i++) x(i)=1.0+0.5*std::sin(i+1.0); return x; }

// ─────────────────────────────────────────────────────────────────────────────
// Output
// ─────────────────────────────────────────────────────────────────────────────

static void header() {
    printf("%-30s %6s %10s %7s %14s\n", "# scenario", "N", "tape_nodes", "iters", "t/call(s)");
}

// Prints one row; returns false if t exceeds skip_thresh (caller should stop the N loop).
static bool row(const char* name, int N, size_t tape_sz, int iters, double t,
                double skip_thresh = std::numeric_limits<double>::infinity()) {
    printf("%-30s %6d %10zu %7d %14.4e\n", name, N, tape_sz, iters, t);
    fflush(stdout);
    return t < skip_thresh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-operation benchmark helpers
// ─────────────────────────────────────────────────────────────────────────────

template<typename Fn>
bool bench_grad(const char* name, int N, Eigen::VectorXd xd, Fn fn,
                double skip = std::numeric_limits<double>::infinity())
{
    Tape<double> tp; { auto xp = record_matrix(xd,tp); fn(xp); }  // probe
    size_t ts = tp.size();
    auto [t, iters] = run([&]{
        Tape<double> t0; auto x = record_matrix(xd, t0);
        auto [g] = gradient(fn(x), x);
        g_sink += g(0);
    });
    return row(name, N, ts, iters, t, skip);
}

// sparse_jacobian: each output in F depends on ~k inputs.
// fn returns std::vector<V1>.
template<typename Fn>
bool bench_sparse_jacobian(const char* name, int N, Eigen::VectorXd xd, Fn fn,
                           double skip = std::numeric_limits<double>::infinity())
{
    Tape<double> tp; { auto xp = record_matrix(xd,tp); fn(xp); }
    size_t ts = tp.size();
    auto [t, iters] = run([&]{
        Tape<double> t0; auto x = record_matrix(xd, t0);
        auto F = fn(x);
        auto J = sparse_jacobian<double>(F, x);
        g_sink += J.coeff(0,0);
    });
    return row(name, N, ts, iters, t, skip);
}

template<typename Fn>
bool bench_hessian(const char* name, int N, Eigen::VectorXd xd, Fn fn,
                   double skip = std::numeric_limits<double>::infinity())
{
    Tape<double> t1p; Tape<V1> t2p; { auto xp = record_matrix(xd,t1p,t2p); fn(xp); }
    size_t ts = t2p.size();
    auto [t, iters] = run([&]{
        Tape<double> t1; Tape<V1> t2; auto x = record_matrix(xd, t1, t2);
        auto H = hessian(fn(x), x);
        g_sink += H(0,0);
    });
    return row(name, N, ts, iters, t, skip);
}

template<typename Fn>
bool bench_sparse_hessian(const char* name, int N, Eigen::VectorXd xd, Fn fn,
                          double skip = std::numeric_limits<double>::infinity())
{
    Tape<double> t1p; Tape<V1> t2p; { auto xp = record_matrix(xd,t1p,t2p); fn(xp); }
    size_t ts = t2p.size();
    auto [t, iters] = run([&]{
        Tape<double> t1; Tape<V1> t2; auto x = record_matrix(xd, t1, t2);
        auto H = sparse_hessian(fn(x), x);
        g_sink += H.coeff(0,0);
    });
    return row(name, N, ts, iters, t, skip);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    header();

    // ── grad() ───────────────────────────────────────────────────────────────
    printf("# --- grad() : first derivative, dense backward pass\n");
    for (int N = 4; N <= 1024; N *= 2)
        bench_grad("spring/grad",     2*N, spring_x(N), [N](auto& x){ return spring_energy(x,N); });
    printf("\n");
    for (int N : {3,4,6,8,12,16,24})
        bench_grad("llt/grad",        N,   llt_x(N),   [](auto& x)  { return llt_func(x);        });
    printf("\n");
    for (int N = 4; N <= 1024; N *= 2)
        bench_grad("rosenbrock/grad", N,   rosen_x(N), [](auto& x)  { return rosenbrock(x);      });
    printf("\n");
    for (int N = 4; N <= 512; N *= 2)
        bench_grad("trig/grad",       N,   trig_x(N),  [N](auto& x){ return trig_func(x,N);      });
    printf("\n");
    for (int N = 4; N <= 512; N *= 2)
        bench_grad("ratio/grad",      N,   ratio_x(N), [N](auto& x){ return ratio_func(x,N);     });
    printf("\n");
    // Random graph: irregular sparsity, large N
    for (int N : {100, 500, 1000, 5000, 10000}) {
        auto g = make_random_graph(N);
        bench_grad("rand_spring/grad", 2*N, g.x, [&g](auto& x){ return random_spring_energy(x, g.edges); });
    }
    printf("\n");

    // ── sparse_jacobian() ─────────────────────────────────────────────────────
    // stencil: each of N outputs depends on 2 inputs → bidiagonal J (2 nz/row).
    // This exercises sparse_grad in its intended use case.
    printf("# --- sparse_jacobian() : bidiagonal stencil, 2 nonzeros per row\n");
    { bool ok=true; for (int N=4; N<=1024 && ok; N*=2)
        ok = bench_sparse_jacobian("stencil/sparse_jacobian", N, trig_x(N),
                                   [N](auto& x){ return stencil_func(x,N); }, 0.5); }
    printf("\n");

    // ── hessian() dense — auto-skip above 0.5 s/call ─────────────────────────
    printf("# --- hessian() : second derivative, dense (skips when >0.5s/call)\n");
    { bool ok=true; for (int N=4; N<=512 && ok; N*=2)
        ok = bench_hessian("spring/hessian",     2*N, spring_x(N), [N](auto& x){ return spring_energy(x,N); }, 0.5); }
    printf("\n");
    { bool ok=true; for (int N : {3,4,6,8,12,16}) if (ok)
        ok = bench_hessian("llt/hessian",        N,   llt_x(N),   [](auto& x)  { return llt_func(x);        }, 0.5); }
    printf("\n");
    { bool ok=true; for (int N=4; N<=512 && ok; N*=2)
        ok = bench_hessian("rosenbrock/hessian", N,   rosen_x(N), [](auto& x)  { return rosenbrock(x);      }, 0.5); }
    printf("\n");
    { bool ok=true; for (int N=4; N<=256 && ok; N*=2)
        ok = bench_hessian("trig/hessian",       N,   trig_x(N),  [N](auto& x){ return trig_func(x,N);      }, 0.5); }
    printf("\n");
    { bool ok=true; for (int N=4; N<=256 && ok; N*=2)
        ok = bench_hessian("ratio/hessian",      N,   ratio_x(N), [N](auto& x){ return ratio_func(x,N);     }, 0.5); }
    printf("\n");

    // ── sparse_hessian() — auto-skip above 0.5 s/call ────────────────────────
    // llt omitted: its Hessian is fully dense (all vars coupled through A).
    printf("# --- sparse_hessian() : second derivative, sparse (skips when >0.5s/call)\n");
    { bool ok=true; for (int N=4; N<=1024 && ok; N*=2)
        ok = bench_sparse_hessian("spring/sparse_hessian",     2*N, spring_x(N), [N](auto& x){ return spring_energy(x,N); }, 0.5); }
    printf("\n");
    { bool ok=true; for (int N=4; N<=512 && ok; N*=2)
        ok = bench_sparse_hessian("rosenbrock/sparse_hessian", N,   rosen_x(N), [](auto& x)  { return rosenbrock(x);      }, 0.5); }
    printf("\n");
    { bool ok=true; for (int N=4; N<=512 && ok; N*=2)
        ok = bench_sparse_hessian("trig/sparse_hessian",       N,   trig_x(N),  [N](auto& x){ return trig_func(x,N);      }, 0.5); }
    printf("\n");
    { bool ok=true; for (int N=4; N<=512 && ok; N*=2)
        ok = bench_sparse_hessian("ratio/sparse_hessian",      N,   ratio_x(N), [N](auto& x){ return ratio_func(x,N);     }, 0.5); }
    printf("\n");
    // Random graph: irregular (non-banded) sparsity, large N, ~4 edges/node
    { bool ok=true;
      for (int N : {100, 500, 1000, 5000, 10000}) if (ok) {
          auto g = make_random_graph(N);
          ok = bench_sparse_hessian("rand_spring/sparse_hessian", 2*N, g.x,
                                    [&g](auto& x){ return random_spring_energy(x, g.edges); }, 0.5);
      }
    }
    printf("\n");

    return 0;
}
