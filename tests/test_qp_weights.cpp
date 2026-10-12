// qp_weights_measurement, the quasiparticle residue and N-phonon weights.
//
// 1. Bookkeeping, on a chain with all updates at a generic p: the test keeps its own sums of s,
//    s delta_{N_ext,N}, s N_ext and s delta_{N_ext,0} U P U^T / |tr P| for the same samples, and the
//    estimator's means must be their ratios. Exact identities on top: sum_N Z_N = 1, <N> = sum_N N Z_N,
//    tr R = Z_0 in both bases, R symmetric, R_band = U(p)^T R_orb U(p). As in test_green_trace, every
//    third measurement is taken on the diagram with its band matrix negated, to exercise the sign.
// 2. Cubic symmetry at p = 0: R_orb = (Z_0/3) I within the jackknife errors. Also reports how far the
//    head's k drifts from p = 0 in the N = 0 sector (rounding of the external-line updates), the reason R
//    is accumulated in the orbital basis.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "test_common.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "measurements/qp_weights.hpp"

struct Chain {
    test::Diagram d;
    add_int_ph_update addi;  rm_int_ph_update rmi;
    add_ext_ph_update adde;  rm_ext_ph_update rme;
    swp_ph_update swp;       mv_tau_update mv;
    chg_tau_update ch;       chg_ph_momentum chw;
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 7};

    Chain(unsigned long long seed, std::array<double, 3> p)
        : d(seed, p, 24, 8),
          addi(&d.cfg, &d.rng), rmi(&d.cfg, &d.rng), adde(&d.cfg, &d.rng), rme(&d.cfg, &d.rng),
          swp(&d.cfg, &d.rng), mv(&d.cfg, &d.rng), ch(&d.cfg, &d.rng), chw(&d.cfg, &d.rng) {}

    void step(){
        const int w {pick(d.rng)};
        double r {-1.};
        switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                     case 3: r = rme.attempt(); break; case 4: r = swp.attempt(); break; case 5: r = mv.attempt(); break;
                     case 6: r = ch.attempt(); break; default: r = chw.attempt(); }
        if (r > 0. && u(d.rng) < r) {
            switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                         case 6: ch.accept(); break; default: chw.accept(); }
        }
    }
};

static void bookkeeping(test::Checks & check){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    const long n_steps {600000};

    // first pass: the estimator, on samples with every third band matrix negated
    std::uint64_t n_used {0};               // samples in full batches: the only ones the jackknife sees
    qp_weights_measurement::result r;
    std::uint64_t n_measured {0}, n_negative {0};
    int n_max {0};
    {
        Chain c {4242ULL, p};
        diagram_cfg * cfg {&c.d.cfg};
        qp_weights_measurement Q {cfg, p, 0.};
        for (long step {0}; step < n_steps; ++step) {
            c.step();
            const bool flip {step % 3 == 0};
            if (flip) { cfg->diagram_head->right_component *= -1.; }
            Q.measure();
            if (flip) { cfg->diagram_head->right_component *= -1.; }
        }
        r = Q.normalised();
        n_used = Q.acc.batches().size() * Q.acc.batch_count();
        n_measured = Q.n_measured;
        n_negative = Q.n_negative;
        n_max = Q.n_max;
    }

    // second pass: the same deterministic chain, summed independently over the samples in full batches
    double sum_s {0.}, sum_sN {0.};
    std::vector<double> sum_hist(static_cast<std::size_t>(n_max + 1), 0.);
    Eigen::Matrix3d sum_R {Eigen::Matrix3d::Zero()};
    {
        Chain c {4242ULL, p};
        diagram_cfg * cfg {&c.d.cfg};
        for (long step {0}; step < static_cast<long>(n_used); ++step) {
            c.step();
            const Eigen::Matrix3d P {(step % 3 == 0 ? -1. : 1.) * cfg->diagram_head->right_component};
            const double s {P.trace() < 0. ? -1. : 1.};
            const int N {cfg->external_ph_manager->current_length / 2};
            sum_s += s;
            sum_sN += s * N;
            sum_hist[static_cast<std::size_t>(N)] += s;
            if (N == 0) {
                const Eigen::Matrix3d & U {cfg->diagram_head->baseWF};
                const Eigen::Matrix3d M {U * P * U.transpose() / std::abs(P.trace())};
                sum_R += 0.5 * (M + M.transpose());
            }
        }
    }

    double dev_Z {0.}, total {0.}, mean_N {0.};
    for (int N {0}; N <= n_max; ++N) {
        const auto n {static_cast<std::size_t>(N)};
        dev_Z = std::max(dev_Z, std::abs(static_cast<double>(r.Z[n]) - sum_hist[n] / sum_s));
        total += static_cast<double>(r.Z[n]);
        mean_N += N * static_cast<double>(r.Z[n]);
    }
    const double dev_N {std::abs(static_cast<double>(r.mean_N) - sum_sN / sum_s)};
    const double dev_R {(r.R_orb - sum_R / sum_s).cwiseAbs().maxCoeff()};
    check(dev_Z < 1e-12 && dev_N < 1e-12 && dev_R < 1e-12 && n_measured == static_cast<std::uint64_t>(n_steps) && n_negative > 0,
          "Z_N, <N> and R_orb are the ratios of the test's own sums over the %llu samples in full batches (%llu measured, %llu negative): deviations %.1e %.1e %.1e",
          static_cast<unsigned long long>(n_used), static_cast<unsigned long long>(n_measured),
          static_cast<unsigned long long>(n_negative), dev_Z, dev_N, dev_R);

    const double Z0 {static_cast<double>(r.Z[0])};
    const Eigen::Matrix3d U_p {weight::LKMatrix::diagonalizeLKHamiltonian(p).block<3,3>(1,0)};
    check(std::abs(total - 1.) < 1e-12 && std::abs(mean_N - static_cast<double>(r.mean_N)) < 1e-12,
          "sum_N Z_N = 1 (%.1e) and <N> = sum_N N Z_N (%.1e); Z_0 = %.4f, Z_1 = %.4f, <N> = %.4f",
          std::abs(total - 1.), std::abs(mean_N - static_cast<double>(r.mean_N)), Z0, static_cast<double>(r.Z[1]),
          static_cast<double>(r.mean_N));
    check(std::abs(r.R_orb.trace() - Z0) < 1e-12 && std::abs(r.R_band.trace() - Z0) < 1e-12
          && (r.R_orb - r.R_orb.transpose()).cwiseAbs().maxCoeff() < 1e-15
          && (r.R_band - U_p.transpose() * r.R_orb * U_p).cwiseAbs().maxCoeff() < 1e-12,
          "tr R = Z_0 in both bases (%.1e, %.1e), R symmetric, R_band = U(p)^T R_orb U(p)",
          std::abs(r.R_orb.trace() - Z0), std::abs(r.R_band.trace() - Z0));
}

static void cubic_symmetry(test::Checks & check){
    const std::array<double, 3> p {0., 0., 0.};
    Chain c {777ULL, p};
    diagram_cfg * cfg {&c.d.cfg};
    qp_weights_measurement Q {cfg, p, 0.};

    double drift {0.};
    for (long step {0}; step < 2000000; ++step) {
        c.step();
        Q.measure();
        if (cfg->external_ph_manager->current_length == 0) {
            const std::array<double, 3> & k {cfg->diagram_head->k};
            drift = std::max({drift, std::abs(k[0]), std::abs(k[1]), std::abs(k[2])});
        }
    }

    const qp_weights_measurement::result r {Q.normalised()};
    const double Z0 {static_cast<double>(r.Z[0])};
    const Eigen::Matrix3d dev {r.R_orb - (Z0 / 3.) * Eigen::Matrix3d::Identity()};
    double worst_pull {0.};
    for (int i {0}; i < 3; ++i) {
        for (int j {0}; j < 3; ++j) {
            // the diagonal of R_orb - Z_0/3 is correlated with Z_0; its error is bounded by R_orb's own
            const double e {std::max(r.R_orb_error(i, j), 1e-300)};
            worst_pull = std::max(worst_pull, std::abs(dev(i, j)) / e);
        }
    }
    check(worst_pull < 5.,
          "p = 0: R_orb = (Z_0/3) I within errors, worst pull %.2f sigma (Z_0 = %.4f +- %.4f, R_xx = %.4f +- %.4f)",
          worst_pull, Z0, static_cast<double>(r.Z_error[0]), r.R_orb(0,0), r.R_orb_error(0,0));
    std::printf("  [info] largest |k_head| component in the N = 0 sector at p = 0: %.1e\n", drift);
}

int main(){
    test::Checks check {"qp_weights_measurement"};
    test::setLK(test::AlAs);
    bookkeeping(check);
    cubic_symmetry(check);
    return check.exit_code();
}
