// Micro-benchmark: free LK propagator G0(k, tau) = exp(-tau H(k)) (mu = 0), numerical diagonalization against the
// closed forms of latex_notes/free_green_function.tex.
//
//     make -C tests bench
//     taskset -c 4 build/tests/bench_lk_green [A B C tau]
//
// Methods (each returns the full 3x3 G0, which is gauge independent):
//   production        weight::LKMatrix::diagonalizeLKHamiltonian (canonical direction, wedge, iterative QR, sign
//                     gauge), then U diag(e^{-tau eps}) U^T
//   eigen_qr          Eigen::SelfAdjointEigenSolver::compute on the raw H(k) (iterative QR), U diag U^T
//   eigen_direct      Eigen::SelfAdjointEigenSolver::computeDirect (Cardano + cross products), U diag U^T
//   closed_sylvester  trigonometric Cardano, P_n = prod_{m != n} (H - eps_m)/(eps_n - eps_m)
//   closed_rank1      trigonometric Cardano, P_n = u u^T / |u|^2 with u_i = k_i / (eps_n - lambda_i)
// plus the eigenvalue-only variants. Accuracy is measured against a long double reference
// (SelfAdjointEigenSolver<Matrix<long double, 3, 3>>), on generic directions and on directions at an angle delta from
// [111] and [100], where bands are (nearly) degenerate.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "comp_method/weight_computation.hpp"

namespace {
    using Vec = std::array<double, 3>;
    using M3 = Eigen::Matrix3d;
    using M3l = Eigen::Matrix<long double, 3, 3>;

    double A {4.681}, B {1.019}, C {5.498}, TAU {1.};

    // eigenvalues (ascending) and G0
    struct Out {
        Eigen::Vector3d eps;
        M3 G;
    };

    M3 hamiltonian(const Vec & k){
        const double k2 {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};
        M3 H;
        for (int i {0}; i < 3; ++i) {
            for (int j {0}; j < 3; ++j) {
                H(i, j) = i == j ? A*k[i]*k[i] + B*(k2 - k[i]*k[i]) : C*k[i]*k[j];
            }
        }
        return H;
    }

    M3 fromBands(const Eigen::Vector3d & eps, const M3 & U){
        const Eigen::Vector3d e {(-TAU * eps.array()).exp()};
        return U * e.asDiagonal() * U.transpose();
    }

    // ---- methods ------------------------------------------------------------------------------------------------

    Out production(const Vec & k){
        const Eigen::Matrix<double, 4, 3> r {weight::LKMatrix::diagonalizeLKHamiltonian(k)};
        const double k2 {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};
        const Eigen::Vector3d eps {k2 * r.row(0).transpose()};
        return {eps, fromBands(eps, r.bottomRows<3>())};
    }

    Out eigenQR(const Vec & k){
        Eigen::SelfAdjointEigenSolver<M3> es;
        es.compute(hamiltonian(k));
        return {es.eigenvalues(), fromBands(es.eigenvalues(), es.eigenvectors())};
    }

    Out eigenDirect(const Vec & k){
        Eigen::SelfAdjointEigenSolver<M3> es;
        es.computeDirect(hamiltonian(k));
        return {es.eigenvalues(), fromBands(es.eigenvalues(), es.eigenvectors())};
    }

    // trigonometric Cardano for a real symmetric 3x3, ascending
    Eigen::Vector3d cardano(const M3 & H){
        const double q {H.trace() / 3.};
        const M3 Hs {H - q * M3::Identity()};
        const double p2 {(Hs * Hs).trace() / 6.};
        if (p2 == 0.) { return Eigen::Vector3d::Constant(q); }
        const double p {std::sqrt(p2)};
        const double r {std::clamp((Hs / p).determinant() / 2., -1., 1.)};
        const double phi {std::acos(r) / 3.};
        const double e1 {q + 2.*p*std::cos(phi)};
        const double e3 {q + 2.*p*std::cos(phi + 2.*std::numbers::pi/3.)};
        return {e3, 3.*q - e1 - e3, e1};
    }

    Out closedSylvester(const Vec & k){
        const M3 H {hamiltonian(k)};
        const Eigen::Vector3d e {cardano(H)};
        const M3 I {M3::Identity()};
        const M3 M0 {H - e(0)*I}, M1 {H - e(1)*I}, M2 {H - e(2)*I};
        const M3 G {std::exp(-TAU*e(0)) * (M1 * M2) / ((e(0) - e(1)) * (e(0) - e(2)))
                  + std::exp(-TAU*e(1)) * (M0 * M2) / ((e(1) - e(0)) * (e(1) - e(2)))
                  + std::exp(-TAU*e(2)) * (M0 * M1) / ((e(2) - e(0)) * (e(2) - e(1)))};
        return {e, G};
    }

    Out closedRank1(const Vec & k){
        const M3 H {hamiltonian(k)};
        const Eigen::Vector3d e {cardano(H)};
        const double k2 {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};
        const double delta {A - B - C};
        const Eigen::Vector3d kv {k[0], k[1], k[2]};
        const Eigen::Vector3d lambda {(B*k2 + delta * kv.array().square()).matrix()};
        M3 G {M3::Zero()};
        for (int n {0}; n < 3; ++n) {
            const Eigen::Vector3d u {(kv.array() / (e(n) - lambda.array())).matrix()};
            G += std::exp(-TAU*e(n)) * (u * u.transpose()) / u.squaredNorm();
        }
        return {e, G};
    }

    // eigenvalue-only variants
    Eigen::Vector3d valsProduction(const Vec & k){
        const double k2 {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};
        return k2 * weight::LKMatrix::diagonalizeLKHamiltonianEigenval(k).transpose();
    }
    Eigen::Vector3d valsQR(const Vec & k){
        Eigen::SelfAdjointEigenSolver<M3> es;
        es.compute(hamiltonian(k), Eigen::EigenvaluesOnly);
        return es.eigenvalues();
    }
    Eigen::Vector3d valsDirect(const Vec & k){
        Eigen::SelfAdjointEigenSolver<M3> es;
        es.computeDirect(hamiltonian(k), Eigen::EigenvaluesOnly);
        return es.eigenvalues();
    }
    Eigen::Vector3d valsCardano(const Vec & k){ return cardano(hamiltonian(k)); }

    // long double reference
    struct Ref {
        Eigen::Matrix<long double, 3, 1> eps;
        M3l G;
    };
    Ref reference(const Vec & k){
        const M3 H {hamiltonian(k)};
        Eigen::SelfAdjointEigenSolver<M3l> es;
        es.compute(H.cast<long double>());
        Eigen::Matrix<long double, 3, 1> e;
        for (int n {0}; n < 3; ++n) { e(n) = std::exp(-static_cast<long double>(TAU) * es.eigenvalues()(n)); }
        return {es.eigenvalues(), es.eigenvectors() * e.asDiagonal() * es.eigenvectors().transpose()};
    }

    // ---- inputs -------------------------------------------------------------------------------------------------

    Vec randomUnit(std::mt19937_64 & rng){
        std::normal_distribution<double> n;
        Vec v {n(rng), n(rng), n(rng)};
        const double s {std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])};
        return {v[0]/s, v[1]/s, v[2]/s};
    }

    // unit vector at angle delta from the unit vector d, in a random perpendicular direction
    Vec nearDirection(const Vec & d, double delta, std::mt19937_64 & rng){
        Vec t {randomUnit(rng)};
        const double dot {t[0]*d[0] + t[1]*d[1] + t[2]*d[2]};
        for (int i {0}; i < 3; ++i) { t[i] -= dot * d[i]; }
        const double tn {std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2])};
        Vec v;
        for (int i {0}; i < 3; ++i) { v[i] = std::cos(delta) * d[i] + std::sin(delta) * t[i] / tn; }
        return v;
    }

    std::vector<Vec> momenta(std::size_t n, std::mt19937_64 & rng, const std::function<Vec(std::mt19937_64 &)> & dir){
        std::uniform_real_distribution<double> mag(0.1, 3.);
        std::vector<Vec> ks(n);
        for (Vec & k : ks) {
            const Vec u {dir(rng)};
            const double m {mag(rng)};
            k = {m*u[0], m*u[1], m*u[2]};
        }
        return ks;
    }

    double median(std::vector<double> v){
        if (v.empty()) { return std::numeric_limits<double>::quiet_NaN(); }
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size()/2), v.end());
        return v[v.size()/2];
    }

    // ---- timing -------------------------------------------------------------------------------------------------

    volatile double sink {0.};

    template <typename F>
    void timeIt(const char * name, const std::vector<Vec> & ks, F f){
        constexpr int reps {7};
        std::vector<double> ns;
        double check {0.};
        for (int r {-1}; r < reps; ++r) {      // r = -1: warm-up
            const auto t0 {std::chrono::steady_clock::now()};
            for (const Vec & k : ks) { check += f(k); }
            const auto t1 {std::chrono::steady_clock::now()};
            if (r >= 0) { ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(ks.size())); }
        }
        sink = sink + check;
        std::printf("  %-26s %9.1f %9.1f\n", name, *std::min_element(ns.begin(), ns.end()), median(ns));
    }

    // ---- accuracy -----------------------------------------------------------------------------------------------

    struct Stats {
        std::vector<double> g, gap;
        int bad {0};
    };

    void accuracy(const char * set, const std::vector<Vec> & ks){
        const std::array<const char *, 5> names {"production", "eigen_qr", "eigen_direct", "closed_sylvester", "closed_rank1"};
        const std::array<Out (*)(const Vec &), 5> fs {production, eigenQR, eigenDirect, closedSylvester, closedRank1};
        std::array<Stats, 5> st;
        for (const Vec & k : ks) {
            const Ref ref {reference(k)};
            const long double gmax {ref.G.cwiseAbs().maxCoeff()};
            const long double gap_ref {std::min(ref.eps(1) - ref.eps(0), ref.eps(2) - ref.eps(1))};
            for (std::size_t m {0}; m < fs.size(); ++m) {
                const Out o {fs[m](k)};
                const double eg {static_cast<double>((o.G.cast<long double>() - ref.G).cwiseAbs().maxCoeff() / gmax)};
                if (!std::isfinite(eg)) { ++st[m].bad; continue; }
                st[m].g.push_back(eg);
                Eigen::Vector3d e {o.eps};
                std::sort(e.data(), e.data() + 3);
                const long double gap {static_cast<long double>(std::min(e(1) - e(0), e(2) - e(1)))};
                if (gap_ref > 1e-13L * std::abs(ref.eps(2))) {     // a resolvable splitting only
                    st[m].gap.push_back(static_cast<double>(std::abs(gap - gap_ref) / gap_ref));
                }
            }
        }
        std::printf("  %s\n", set);
        for (std::size_t m {0}; m < fs.size(); ++m) {
            const auto mx {[](const std::vector<double> & v){ return v.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                                               : *std::max_element(v.begin(), v.end()); }};
            std::printf("    %-18s G0 err max %9.2e med %9.2e   gap err max %9.2e med %9.2e   non-finite %d\n", names[m],
                        mx(st[m].g), median(st[m].g), mx(st[m].gap), median(st[m].gap), st[m].bad);
        }
    }
}

int main(int argc, char ** argv){
    if (argc > 1) { A = std::atof(argv[1]); }
    if (argc > 2) { B = std::atof(argv[2]); }
    if (argc > 3) { C = std::atof(argv[3]); }
    if (argc > 4) { TAU = std::atof(argv[4]); }
    if (C == 0.) { std::printf("C = 0: the code takes the exact diagonal shortcut, nothing to compare\n"); return 0; }
    weight::LKMatrix::setLKParameters(A, B, C);

    std::printf("bench_lk_green   A = %g  B = %g  C = %g  (Delta = A - B - C = %g)  tau = %g   compiler %s%s\n", A, B, C,
                A - B - C, TAU, __VERSION__,
#if defined(NDEBUG) && defined(__OPTIMIZE__)
                "  (optimised, NDEBUG)"
#else
                "  (WARNING: not an optimised NDEBUG build)"
#endif
                );

    std::mt19937_64 rng {20261105};
    const std::vector<Vec> ks {momenta(1'000'000, rng, randomUnit)};

    std::printf("\ntiming, %zu random momenta, ns per call:          min    median\n", ks.size());
    std::printf(" G0(k, tau), full matrix\n");
    const auto sumG {[](Out (*f)(const Vec &)){ return [f](const Vec & k){ return f(k).G.sum(); }; }};
    timeIt("production", ks, sumG(production));
    timeIt("eigen_qr", ks, sumG(eigenQR));
    timeIt("eigen_direct", ks, sumG(eigenDirect));
    timeIt("closed_sylvester", ks, sumG(closedSylvester));
    timeIt("closed_rank1", ks, sumG(closedRank1));
    std::printf(" eigenvalues only\n");
    const auto sumE {[](Eigen::Vector3d (*f)(const Vec &)){ return [f](const Vec & k){ return f(k).sum(); }; }};
    timeIt("production (Eigenval)", ks, sumE(valsProduction));
    timeIt("eigen_qr", ks, sumE(valsQR));
    timeIt("eigen_direct", ks, sumE(valsDirect));
    timeIt("cardano", ks, sumE(valsCardano));
    timeIt("hamiltonian only (floor)", ks, [](const Vec & k){ return hamiltonian(k).sum(); });

    std::printf("\naccuracy against long double (relative): G0 = max|G - G_ref| / max|G_ref|, gap = smallest band splitting\n");
    constexpr std::size_t n_acc {10'000};
    accuracy("generic directions", momenta(n_acc, rng, randomUnit));
    const double s3 {1./std::sqrt(3.)};
    const Vec d111 {s3, s3, s3}, d100 {1., 0., 0.};
    for (const double delta : {1e-2, 1e-4, 1e-6, 1e-8}) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "angle %.0e from [111]", delta);
        accuracy(buf, momenta(n_acc, rng, [&](std::mt19937_64 & g){ return nearDirection(d111, delta, g); }));
        std::snprintf(buf, sizeof buf, "angle %.0e from [100]", delta);
        accuracy(buf, momenta(n_acc, rng, [&](std::mt19937_64 & g){ return nearDirection(d100, delta, g); }));
    }
    accuracy("exactly [111] (gap: degenerate pair, not shown)", momenta(n_acc, rng, [&](std::mt19937_64 &){ return d111; }));
    accuracy("exactly [100] (gap: degenerate pair, not shown)", momenta(n_acc, rng, [&](std::mt19937_64 &){ return d100; }));
    return 0;
}
