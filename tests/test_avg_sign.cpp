// avg_sign_measurement, the average sign <s> = sum W / sum |W>:
//   1. C = 0 (no sign problem): <s> = 1 exactly, error 0, no negative samples;
//   2. AlAs three-band parameters (C != 0) at fixed tau_D, all length-preserving updates: measured on the same
//      samples as gs_energy, <s> equals the mean of gs_energy's s component (same batches, same jackknife) and
//      1 - 2 n_negative / n_measured, and the two estimators count the same negative samples;
//   3. summary(): NaNs before 2 full batches, then exactly normalised()'s value and error.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/jackknife.hpp>
#include "test_common.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "measurements/avg_sign.hpp"
#include "measurements/gs_energy.hpp"

// a fixed-length chain with every length-preserving update, measuring S (and E) every 10 steps
struct Chain {
    test::Diagram d;
    add_int_ph_update addi;  rm_int_ph_update rmi;
    add_ext_ph_update adde;  rm_ext_ph_update rme;
    mv_tau_update mv;        swp_ph_update swp;
    chg_ph_momentum chw;
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 6};

    Chain(unsigned long long seed, double L, double omega, double eps)
        : d(seed, {0., 0., 0.}, 400, 16, L, 0., omega, eps),
          addi(&d.cfg, &d.rng), rmi(&d.cfg, &d.rng), adde(&d.cfg, &d.rng), rme(&d.cfg, &d.rng),
          mv(&d.cfg, &d.rng), swp(&d.cfg, &d.rng), chw(&d.cfg, &d.rng) {}

    void step(){
        const int w {pick(d.rng)};
        double r {-1.};
        switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                     case 3: r = rme.attempt(); break; case 4: r = mv.attempt(); break; case 5: r = swp.attempt(); break;
                     default: r = chw.attempt(); }
        const bool acc {r > 0. && u(d.rng) < r};
        switch (w) { case 0: acc ? addi.accept() : addi.reject(); break; case 1: acc ? rmi.accept() : rmi.reject(); break;
                     case 2: acc ? adde.accept() : adde.reject(); break; case 3: acc ? rme.accept() : rme.reject(); break;
                     case 4: acc ? mv.accept() : mv.reject(); break; case 5: acc ? swp.accept() : swp.reject(); break;
                     default: acc ? chw.accept() : chw.reject(); }
    }
};

int main(){
    test::Checks check {"avg_sign_measurement (average sign)"};

    // 1. C = 0
    {
        test::setLK({1., 1., 0.});
        Chain c {7ULL, 10., 1., 1.};
        avg_sign_measurement S {&c.d.cfg, 0., true, 16};
        const std::vector<double> before {S.summary()};
        for (int i {0}; i < 200000; ++i) { c.step(); if (i % 10 == 0) { S.measure(); } }
        const auto r {S.normalised()};
        const std::vector<double> after {S.summary()};
        check(r.sign == 1.L && r.sign_error == 0.L && S.n_negative == 0 && S.n_measured == 20000
              && std::isnan(before[0]) && std::isnan(before[1]) && after[0] == 1. && after[1] == 0.,
              "C = 0: <s> = %.6Lf +- %.1Le, %llu negative of %llu; summary() NaN before 2 batches, then normalised()",
              r.sign, r.sign_error, static_cast<unsigned long long>(S.n_negative), static_cast<unsigned long long>(S.n_measured));
    }

    // 2. C != 0: the same samples as gs_energy
    {
        test::setLK(test::AlAs);
        Chain c {11ULL, 60., 1., 0.5};
        avg_sign_measurement S {&c.d.cfg, 0., true, 16};
        gs_energy_measurement E {&c.d.cfg, {0., 0., 0.}, 0., true, 16};
        for (int i {0}; i < 2000000; ++i) { c.step(); if (i % 10 == 0) { S.measure(); E.measure(); } }
        const auto r {S.normalised()};
        const auto jk {simplemc::jackknife([](const Eigen::VectorXd & x) -> Eigen::VectorXd { return Eigen::VectorXd::Constant(1, x(1)); }, E.acc)};
        const double s_gs {jk.naive_mean()(0)}, err_gs {jk.stderror()(0)};
        const double s_count {1. - 2. * static_cast<double>(S.n_negative) / static_cast<double>(S.n_measured)};
        const std::vector<double> sum {S.summary()};
        const bool same {std::abs(static_cast<double>(r.sign) - s_gs) < 1e-12 && std::abs(static_cast<double>(r.sign_error) - err_gs) < 1e-12
                         && S.n_negative == E.n_negative && S.n_measured == E.n_measured
                         && sum[0] == static_cast<double>(r.sign) && sum[1] == static_cast<double>(r.sign_error)};
        // the counts include the partial batch the jackknife leaves out: equal within the error
        check(same && S.n_negative > 0 && std::abs(static_cast<double>(r.sign) - s_count) < 3. * static_cast<double>(r.sign_error) + 1e-3,
              "C != 0 (AlAs, tau_D = 60, omega = 1, eps = 0.5): <s> = %.5Lf +- %.5Lf = gs_energy's <s> %.5f; 1 - 2 n_neg/n = %.5f "
              "(%llu negative of %llu, the same in both estimators)",
              r.sign, r.sign_error, s_gs, s_count, static_cast<unsigned long long>(S.n_negative),
              static_cast<unsigned long long>(S.n_measured));
    }
    return check.exit_code();
}
