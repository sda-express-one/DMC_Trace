// green_trace_measurement against the trace column of green_func_measurement. Per sample the two are the
// same number - sign(tr P) at the diagram's length, vs the sum of P_nn / |tr P| - so on the same chain,
// with the same normalisation and batches, their means and jackknife errors must agree up to rounding
// (the errors to fewer digits: the jackknife subtracts nearly equal delete-one-batch replicates).
// Checked for the vacuum sector and for all boundary sectors, with and without split_sign.
//
// The chain only ever samples positive traces today, so to exercise the sign bookkeeping every third
// measurement is taken on the diagram with its band matrix negated (restored right after): the signed
// histogram, the positive/negative split and green_func's own P_nm / |tr P| must all follow it. With
// split_sign the parts must then satisfy trace = positive - negative, and the negative part must be the
// (normalised) histogram of exactly the negated samples.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include "test_common.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "measurements/green_func.hpp"
#include "measurements/green_trace.hpp"

static void sector(test::Checks & check, const char * label, int boundary_phonons){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    const int NB {30};
    test::Diagram d {31337ULL, p, 24, 8};
    diagram_cfg * cfg {&d.cfg};
    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 7};

    green_func_measurement G {cfg, p, NB, boundary_phonons};
    green_trace_measurement T {cfg, p, NB, boundary_phonons, false};
    green_trace_measurement S {cfg, p, NB, boundary_phonons, true};
    green_trace_measurement N {cfg, p, NB, boundary_phonons, false};   // only the negated samples

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
        G.measure(); T.measure(); S.measure();
        if (flip) { N.measure(); }
        else {
            // N sees the same order-0 normalisation, but none of the positive samples
            const Eigen::Matrix3d keep {cfg->diagram_head->right_component};
            cfg->diagram_head->right_component.setZero();     // zero trace: skipped by the histogram
            N.measure();
            cfg->diagram_head->right_component = keep;
        }
        if (flip) { cfg->diagram_head->right_component *= -1.; }
    }

    const green_func_measurement::result rg {G.normalised()};
    const green_trace_measurement::result rt {T.normalised()};
    const green_trace_measurement::result rs {S.normalised()};
    const green_trace_measurement::result rn {N.normalised()};

    double scale {0.};
    for (int b {0}; b < NB; ++b) { scale = std::max(scale, static_cast<double>(std::abs(rg.trace[b]))); }

    double dev_mean {0.}, dev_err {0.}, dev_split {0.}, dev_parts {0.}, dev_neg {0.};
    bool hits_equal {true};
    for (int b {0}; b < NB; ++b) {
        const auto bb {static_cast<std::size_t>(b)};
        dev_mean = std::max(dev_mean, static_cast<double>(std::abs(rt.trace[bb] - rg.trace[bb])) / scale);
        dev_err = std::max(dev_err, static_cast<double>(std::abs(rt.trace_error[bb] - rg.trace_error[bb])) / scale);
        dev_split = std::max({dev_split, static_cast<double>(std::abs(rs.trace[bb] - rt.trace[bb])) / scale,
                                         static_cast<double>(std::abs(rs.trace_error[bb] - rt.trace_error[bb])) / scale});
        dev_parts = std::max(dev_parts, static_cast<double>(std::abs(rs.positive[bb] - rs.negative[bb] - rs.trace[bb])) / scale);
        // N holds minus the negated samples' histogram, with the same normalisation
        dev_neg = std::max(dev_neg, static_cast<double>(std::abs(rs.negative[bb] + rn.trace[bb])) / scale);
        hits_equal = hits_equal && T.hits[bb] == G.hits[bb] && S.hits[bb] == G.hits[bb];
    }

    check(dev_mean < 1e-12 && dev_err < 1e-10 && hits_equal && T.n_selected == G.n_selected,
          "%s: trace vs green_func's trace column, %llu selected samples: worst deviation mean %.1e, error %.1e (relative to max |tr G|)",
          label, static_cast<unsigned long long>(T.n_selected), dev_mean, dev_err);
    check(dev_split < 1e-10 && dev_parts < 1e-12 && dev_neg < 1e-12 && S.n_negative == T.n_negative && S.n_negative > 0,
          "%s: split_sign: trace as without split %.1e, positive - negative - trace %.1e, negative part vs negated samples %.1e, %llu negative",
          label, dev_split, dev_parts, dev_neg, static_cast<unsigned long long>(S.n_negative));
}

int main(){
    test::Checks check {"green_trace_measurement vs green_func_measurement"};
    test::setLK(test::AlAs);
    sector(check, "vacuum sector      ", 0);
    sector(check, "all boundary states", -1);
    return check.exit_code();
}
