// green_laguerre_measurement: Laguerre coefficients of g(tau) = tr G(p, tau) e^{mu tau}.
//   1. Basis (measurement::laguerreFunctions): the recurrence against the closed forms for l <= 3,
//      orthonormality of phi_l = sqrt(s) psi_l(s tau) by quadrature, |psi_l| <= 1 and finite up to x = 2000.
//   2. Same samples as green_trace (the test_green_trace chain, every third measurement negated): the
//      coefficients equal Z_0 sum(sign phi_l(tau_D)) / sum(x_0) accumulated by hand, to rounding, and the
//      counters equal green_trace's. The rebuilt tr G agrees with green_trace's histogram bin by bin within
//      the combined errors plus the curvature inside a bin, for nearly every bin. (Its errors are reported, not
//      tested: smaller than the histogram's in the bulk, larger near tau_max, where this chain's g is still large.)
//   3. Order 0 (chg_tau only, as in test_order0): the coefficients agree with the exact ones of
//      sum_n e^{-(E_n - mu) tau} on (0, tau_max] (by quadrature), and the rebuilt tr G with the exact partial
//      sum, with each chain's own jackknife errors.
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
#include "measurements/green_trace.hpp"
#include "measurements/green_laguerre.hpp"
#include "measurements/measurement_utils.hpp"

static void basis(test::Checks & check){
    const int L {60};
    std::vector<double> psi(L + 1);

    double dev_closed {0.};
    for (double x {0.}; x <= 30.; x += 0.37) {
        measurement::laguerreFunctions(x, psi);
        const double e {std::exp(-0.5*x)};
        const double closed[4] {e, (1. - x)*e, (1. - 2.*x + 0.5*x*x)*e, (1. - 3.*x + 1.5*x*x - x*x*x/6.)*e};
        for (int l {0}; l < 4; ++l) { dev_closed = std::max(dev_closed, std::abs(psi[l] - closed[l])); }
    }
    check(dev_closed < 1e-13, "recurrence vs closed forms, l <= 3, x in [0, 30]: max deviation %.1e", dev_closed);

    // orthonormality: int_0^inf psi_l psi_m dx with x = t^2 (L_l oscillates fastest near x = 0, its first zero
    // at ~ 1.4/(4l)), composite Simpson in t on [0, 20] (the tail beyond x = 400 is below e^{-150}); phi_l =
    // sqrt(s) psi_l(s tau) makes the integral the same for any s
    const int N {100000};
    const double h {20. / N};
    std::vector<double> gram((L + 1)*(L + 1), 0.);
    for (int i {0}; i <= N; ++i) {
        const double w {(i == 0 || i == N) ? 1. : (i % 2 ? 4. : 2.)};
        const double t {i * h};
        measurement::laguerreFunctions(t * t, psi);
        for (int l {0}; l <= L; ++l) {
            for (int m {0}; m <= l; ++m) { gram[l*(L + 1) + m] += w * h / 3. * 2. * t * psi[l] * psi[m]; }
        }
    }
    double dev_ortho {0.};
    for (int l {0}; l <= L; ++l) {
        for (int m {0}; m <= l; ++m) { dev_ortho = std::max(dev_ortho, std::abs(gram[l*(L + 1) + m] - (l == m ? 1. : 0.))); }
    }
    check(dev_ortho < 1e-10, "orthonormality of phi_l, l, m <= %d: max |<phi_l, phi_m> - delta_lm| = %.1e", L, dev_ortho);

    double worst {0.};
    bool finite {true};
    for (double x {0.}; x <= 2000.; x += 0.0731) {
        measurement::laguerreFunctions(x, psi);
        for (const double v : psi) { finite = finite && std::isfinite(v); worst = std::max(worst, std::abs(v)); }
    }
    check(finite && worst <= 1. + 1e-12, "|e^{-x/2} L_l(x)| <= 1 and finite for x in [0, 2000], l <= %d: max %.15f", L, worst);
}

static void sameSamples(test::Checks & check, const char * label, int boundary_phonons){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    const int NB {30}, L {40};
    test::Diagram d {31337ULL, p, 24, 8};
    diagram_cfg * cfg {&d.cfg};
    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 7};

    green_trace_measurement T {cfg, p, NB, boundary_phonons, false};
    green_laguerre_measurement G {cfg, p, L, 0., NB, boundary_phonons};

    // by hand: each sample's sign (0 if not selected), tau_D and order-0 indicator; summed at the end over the
    // samples of the accumulator's complete batches (the ones its means are taken over)
    struct Sample { double sign, tau; bool order0; };
    std::vector<Sample> samples;

    for (long step {0}; step < 600000; ++step) {
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

        const bool flip {step % 3 == 0};
        if (flip) { cfg->diagram_head->right_component *= -1.; }
        T.measure(); G.measure();

        const int n_ext {cfg->external_ph_manager->current_length / 2};
        const int n_int {cfg->internal_ph_manager->current_length / 2};
        const bool selected {boundary_phonons < 0 || n_ext == boundary_phonons};
        samples.push_back({selected ? (cfg->diagram_head->right_component.trace() < 0. ? -1. : 1.) : 0.,
                           cfg->current_tau_length, n_int == 0 && n_ext == 0});
        if (flip) { cfg->diagram_head->right_component *= -1.; }
    }

    const green_trace_measurement::result rt {T.normalised()};
    const green_laguerre_measurement::result rg {G.normalised()};
    const long double z0 {measurement::order0Integral(p, cfg->chem_pot, cfg->tau_max)};

    const std::size_t used {G.acc.batches().size() * G.acc.batch_count()};
    std::vector<long double> by_hand(L + 1, 0.0L);
    std::vector<double> psi(L + 1);
    long double n0 {0.0L};
    const double root_s {std::sqrt(G.scale)};
    for (std::size_t i {0}; i < used; ++i) {
        const Sample & x {samples[i]};
        if (x.sign != 0.) {
            measurement::laguerreFunctions(G.scale * x.tau, psi);
            for (int l {0}; l <= L; ++l) { by_hand[l] += x.sign * root_s * psi[l]; }
        }
        if (x.order0) { n0 += 1.0L; }
    }

    double scale {0.}, dev_hand {0.};
    for (int l {0}; l <= L; ++l) { scale = std::max(scale, static_cast<double>(std::abs(rg.coeff[l]))); }
    for (int l {0}; l <= L; ++l) {
        dev_hand = std::max(dev_hand, static_cast<double>(std::abs(rg.coeff[l] - z0 * by_hand[l] / n0)) / scale);
    }
    check(dev_hand < 1e-10 && G.n_selected == T.n_selected && G.n_negative == T.n_negative && G.n_order0 == T.n_order0
              && G.n_negative > 0,
          "%s: coefficients vs hand accumulation: worst deviation %.1e (relative to max |g_l|); counters equal "
          "(%llu selected, %llu negative)", label, dev_hand, static_cast<unsigned long long>(G.n_selected),
          static_cast<unsigned long long>(G.n_negative));

    // rebuilt value at the bin centre vs the bin average: they differ by ~ w^2 g''/24, allowed for with 2 %
    int agree {0};
    std::vector<double> ratio;      // error ratio Laguerre / histogram, for the report
    for (int b {0}; b < NB; ++b) {
        const double diff {static_cast<double>(std::abs(rg.trace[b] - rt.trace[b]))};
        const double sigma {static_cast<double>(std::hypot(rg.trace_error[b], rt.trace_error[b]))};
        if (diff < 4.*sigma + 0.02*std::abs(static_cast<double>(rt.trace[b]))) { ++agree; }
        if (rt.trace_error[b] > 0.0L) { ratio.push_back(static_cast<double>(rg.trace_error[b] / rt.trace_error[b])); }
    }
    std::nth_element(ratio.begin(), ratio.begin() + ratio.size()/2, ratio.end());
    check(agree >= NB - 2, "%s: rebuilt tr G vs green_trace histogram: %d/%d bins agree; median error ratio Laguerre/histogram %.2f",
          label, agree, NB, ratio[ratio.size()/2]);
}

static void order0(test::Checks & check){
    const std::array<double, 3> p {0.155, -0.31, 0.095};      // as in test_order0
    const int CHAINS {4}, NB {40}, L {30};
    const long SAMPLES {3000000};

    double chi2_c {0.}, chi2_t {0.}, worst_pull {0.};
    int dof_c {0}, dof_t {0};
    bool all_order0 {true};
    for (int c {0}; c < CHAINS; ++c) {
        test::Diagram d {9000ULL + static_cast<unsigned long long>(c), p, 4, 4};
        chg_tau_update ch {&d.cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        green_laguerre_measurement G {&d.cfg, p, L, 0., NB, 0};
        for (long s {0}; s < SAMPLES; ++s) {
            const double r {ch.attempt()};
            if (r > 0. && u(d.rng) < r) { ch.accept(); }
            G.measure();
        }
        all_order0 = all_order0 && G.n_order0 == G.acc.count();
        const green_laguerre_measurement::result R {G.normalised()};

        // exact coefficients of sum_n e^{-(E_n - mu) tau} on (0, tau_max], composite Simpson
        const std::array<double, 3> E {d.cfg.diagram_head->electronEnergy()};
        const double mu {d.cfg.chem_pot}, tmax {d.cfg.tau_max}, s {G.scale};
        const int N {40000};
        const double h {tmax / N};
        std::vector<double> exact(L + 1, 0.), psi(L + 1);
        for (int i {0}; i <= N; ++i) {
            const double w {(i == 0 || i == N) ? 1. : (i % 2 ? 4. : 2.)};
            const double tau {i * h};
            double g {0.};
            for (int n {0}; n < 3; ++n) { g += std::exp(-(E[n] - mu) * tau); }
            measurement::laguerreFunctions(s * tau, psi);
            for (int l {0}; l <= L; ++l) { exact[l] += w * h / 3. * g * std::sqrt(s) * psi[l]; }
        }
        for (int l {0}; l <= L; ++l) {
            const double pull {static_cast<double>((R.coeff[l] - exact[l]) / R.coeff_error[l])};
            chi2_c += pull*pull; ++dof_c; worst_pull = std::max(worst_pull, std::abs(pull));
        }
        // the exact partial sum at the grid points, with e^{-mu tau}
        for (int j {0}; j < NB; ++j) {
            const double tau {static_cast<double>(R.tau[j])};
            measurement::laguerreFunctions(s * tau, psi);
            double partial {0.};
            for (int l {0}; l <= L; ++l) { partial += exact[l] * std::sqrt(s) * psi[l]; }
            partial *= std::exp(-mu * tau);
            const double pull {static_cast<double>((R.trace[j] - partial) / R.trace_error[j])};
            chi2_t += pull*pull; ++dof_t; worst_pull = std::max(worst_pull, std::abs(pull));
        }
    }
    check(all_order0, "order 0: every sample of every chain is order 0");
    check(chi2_c/dof_c > 0.6 && chi2_c/dof_c < 1.5, "order 0: coefficients vs exact (quadrature), jackknife errors: chi^2/dof = %.1f/%d = %.3f",
          chi2_c, dof_c, chi2_c/dof_c);
    // the grid values are strongly correlated (they share the coefficients): only a loose band on chi^2/dof
    check(chi2_t/dof_t > 0.2 && chi2_t/dof_t < 3., "order 0: rebuilt tr G vs exact partial sum: chi^2/dof = %.1f/%d = %.3f",
          chi2_t, dof_t, chi2_t/dof_t);
    check(worst_pull < 4.5, "order 0: no outlier: largest |pull| %.2f", worst_pull);
}

int main(){
    test::Checks check {"green_laguerre_measurement"};
    test::setLK(test::AlAs);
    basis(check);
    sameSamples(check, "vacuum sector      ", 0);
    sameSamples(check, "all boundary states", -1);
    order0(check);
    return check.exit_code();
}
