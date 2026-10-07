// The N_modes factor of add_*_ph / rm_*_ph ([N_MODES NORMALIZATION - TO BE VERIFIED] in those files).
//
// The diagrams sum over the phonon mode of every line, so two IDENTICAL modes (same omega, eps) must
// act exactly like a single mode whose every line has twice the weight. With at most one line in the
// diagram, the occupation ratio R = P(order 1) / P(order 0) must therefore double:
//     R(two identical modes) = 2 R(one mode).
// Checked for an internal line (add_int / rm_int / mv_tau / chg_tau) and an external one
// (add_ext / rm_ext / mv_tau / chg_tau). Without the factor (or with it on the wrong side) the
// ratio would come out 1 (or 4). Errors from the spread over independent chains.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "test_common.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"

struct MeanErr { double mean, err; };
static MeanErr spread(const std::vector<double> & x){
    double m {0.}; for (double v : x) { m += v; } m /= static_cast<double>(x.size());
    double s {0.}; for (double v : x) { s += (v - m)*(v - m); }
    return {m, std::sqrt(s / static_cast<double>(x.size() - 1) / static_cast<double>(x.size()))};
}

template <typename U>
static void step(U & up, simplemc::xoshiro256ss & rng){
    std::uniform_real_distribution<double> u {0., 1.};
    const double p {up.attempt()};
    if (p >= 0. && p >= u(rng)) { up.accept(); } else { up.reject(); }
}

// R = P(order 1) / P(order 0) for one chain, with at most one (internal or external) line
template <typename Add, typename Rm>
static double occupation_ratio(std::vector<PhononMode> modes, unsigned long long seed, long steps){
    test::Diagram d {seed, {0.11, -0.07, 0.05}, 2, 2, std::move(modes), 4.0, -0.3};
    diagram_cfg * cfg {&d.cfg};
    Add add {cfg, &d.rng};
    Rm rm {cfg, &d.rng};
    mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};
    std::uniform_int_distribution<int> pick {0, 3};
    double n0 {0.}, n1 {0.};
    for (long s {-100000}; s < steps; ++s) {
        switch (pick(d.rng)) {
            case 0: step(add, d.rng); break;  case 1: step(rm, d.rng); break;
            case 2: step(mv, d.rng); break;   default: step(ch, d.rng); break;
        }
        if (s < 0) { continue; }
        (d.internal.current_length + d.external.current_length == 0 ? n0 : n1) += 1.;
    }
    return n1 / n0;
}

template <typename Add, typename Rm>
static void sector(test::Checks & check, const char * label){
    const PhononMode mode {0.5, 2.0};
    const int CHAINS {8};
    const long STEPS {1500000};
    std::vector<double> one, two;
    for (int c {0}; c < CHAINS; ++c) {
        one.push_back(occupation_ratio<Add, Rm>({mode}, 5100ULL + static_cast<unsigned long long>(c), STEPS));
        two.push_back(occupation_ratio<Add, Rm>({mode, mode}, 6100ULL + static_cast<unsigned long long>(c), STEPS));
    }
    const MeanErr r1 {spread(one)}, r2 {spread(two)};
    const double q {r2.mean / r1.mean};
    const double q_err {q * std::sqrt((r1.err/r1.mean)*(r1.err/r1.mean) + (r2.err/r2.mean)*(r2.err/r2.mean))};
    const double pull {(q - 2.) / q_err};
    check(std::abs(pull) < 4.5,
          "%s: R(1 mode) = %.5f +- %.5f, R(2 identical modes) = %.5f +- %.5f, ratio %.4f +- %.4f vs 2 (%+.1f sig)",
          label, r1.mean, r1.err, r2.mean, r2.err, q, q_err, pull);
}

int main(){
    test::Checks check {"N_modes normalization of add/rm (two identical modes = doubled line weight)"};
    test::setLK(test::AlAs);
    sector<add_int_ph_update, rm_int_ph_update>(check, "internal line");
    sector<add_ext_ph_update, rm_ext_ph_update>(check, "external line");
    return check.exit_code();
}
