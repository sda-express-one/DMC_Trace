// Order 0: with chg_tau as the only update every diagram is the bare propagator, so the Green function
// estimator must reproduce it exactly,
//     G_nm(bin) = delta_nm * (1/w) * int_bin e^{-E_n(p) tau} dtau,
// through the whole chain: chg_tau's proposal and ratio, the order-0 normalisation, the removal of
// e^{mu tau_D}, and the jackknife errors. Checked over several fixed-seed chains (deterministic):
//   - every sample is order 0, and every off-diagonal element is exactly 0;
//   - diagonal components and the trace agree with the exact values, with each chain's own
//     jackknife errors: chi^2/dof close to 1 (errors calibrated) and no outlier pull.
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include "test_common.hpp"
#include "updates/chg_tau.hpp"
#include "measurements/green_func.hpp"

int main(){
    test::Checks check {"Green function estimator at order 0 (chg_tau only)"};
    test::setLK(test::AlAs);

    const std::array<double, 3> p {0.155, -0.31, 0.095};      // generic, non-degenerate
    const int CHAINS {4}, NB {40}, CHECKED_BINS {24};
    const long SAMPLES {3000000};

    bool all_order0 {true};
    double worst_offdiag {0.}, chi2_comp {0.}, chi2_trace {0.}, worst_pull {0.};
    int dof_comp {0}, dof_trace {0};
    for (int c {0}; c < CHAINS; ++c) {
        test::Diagram d {9000ULL + static_cast<unsigned long long>(c), p, 4, 4};
        chg_tau_update ch {&d.cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        green_func_measurement G {&d.cfg, p, NB, 0};
        for (long s {0}; s < SAMPLES; ++s) {
            const double r {ch.attempt()};
            if (r > 0. && u(d.rng) < r) { ch.accept(); }
            G.measure();
        }
        all_order0 = all_order0 && G.n_order0 == G.acc.count();

        const green_func_measurement::result R {G.normalised()};
        const std::array<double, 3> E {d.cfg.diagram_head->electronEnergy()};
        const double w {G.bin_width};
        for (int b {0}; b < NB; ++b) {
            for (int n {0}; n < 3; ++n) {
                for (int m {0}; m < 3; ++m) {
                    if (n != m) { worst_offdiag = std::max(worst_offdiag, static_cast<double>(std::fabs(R.mean[b][3*n + m]))); }
                }
            }
            if (b >= CHECKED_BINS) { continue; }    // the tail bins hold too few samples to test errors
            double exact_trace {0.};
            for (int n {0}; n < 3; ++n) {
                const double exact {(std::exp(-E[n]*b*w) - std::exp(-E[n]*(b + 1)*w)) / (E[n]*w)};
                exact_trace += exact;
                const double pull {static_cast<double>((R.mean[b][4*n] - exact) / R.error[b][4*n])};
                chi2_comp += pull*pull; ++dof_comp; worst_pull = std::max(worst_pull, std::abs(pull));
            }
            const double pull {static_cast<double>((R.trace[b] - exact_trace) / R.trace_error[b])};
            chi2_trace += pull*pull; ++dof_trace; worst_pull = std::max(worst_pull, std::abs(pull));
        }
    }

    check(all_order0, "every sample of every chain is order 0");
    check(worst_offdiag == 0., "off-diagonal elements exactly 0 in every bin (max |G_nm|, n != m: %.1e)", worst_offdiag);
    // dof ~ 288 / 96: a correct estimator gives chi^2/dof within a few times sqrt(2/dof) of 1
    check(chi2_comp/dof_comp > 0.7 && chi2_comp/dof_comp < 1.35,
          "diagonal components vs exact, jackknife errors: chi^2/dof = %.1f/%d = %.3f", chi2_comp, dof_comp, chi2_comp/dof_comp);
    check(chi2_trace/dof_trace > 0.6 && chi2_trace/dof_trace < 1.5,
          "trace vs exact, its own jackknife errors: chi^2/dof = %.1f/%d = %.3f", chi2_trace, dof_trace, chi2_trace/dof_trace);
    check(worst_pull < 4.5, "no outlier: largest |pull| %.2f", worst_pull);
    return check.exit_code();
}
