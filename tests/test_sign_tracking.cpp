// Sign bookkeeping: updates return |ratio| and keep its sign (SignCounter), diagram_cfg tracks the sign
// of the current diagram and counts the steps that end on a negative one.
//
// 1. SignCounter on its own: take() returns |r| and remembers sign(r); accepted() flips the tracked
//    sign on a negative ratio and counts the step if the diagram is then negative; rejected() leaves
//    the sign and counts the step if the diagram is negative.
// 2. A chain with all updates, driven exactly like simplemc's metropolis_kernel (attempt() < 0:
//    impossible -> reject(); u < |r|: accept(); else reject()), with cfg.countSign() after every step
//    as the run loop's on_step callback would:
//    - after every step cfg.current_sign == sign(tr P) of the cached product;
//    - cfg.n_negative == sum of the updates' n_negative (each step belongs to exactly one update);
//    - the sanitizer (which re-derives the sign from the rebuilt trace) finds the state clean.
#include <cstdio>
#include <random>
#include "test_common.hpp"
#include "utils/sign_counter.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "updates/chg_ph_energy.hpp"

static void unit(test::Checks & check){
    test::Diagram d {1ULL, {0.1, 0.2, 0.3}, 4, 4};
    SignCounter s;
    bool ok {d.cfg.current_sign == 1};
    ok = ok && s.take(-0.3) == 0.3 && s.proposed_sign == -1;
    s.accepted(&d.cfg);                                     // + -> -
    ok = ok && d.cfg.current_sign == -1 && s.n_negative == 1;
    s.rejected(&d.cfg);                                     // stays -
    ok = ok && d.cfg.current_sign == -1 && s.n_negative == 2;
    ok = ok && s.take(0.7) == 0.7 && s.proposed_sign == 1;
    s.accepted(&d.cfg);                                     // stays -
    ok = ok && d.cfg.current_sign == -1 && s.n_negative == 3;
    s.take(-2.);
    s.accepted(&d.cfg);                                     // - -> +
    ok = ok && d.cfg.current_sign == 1 && s.n_negative == 3;
    s.rejected(&d.cfg);
    ok = ok && s.n_negative == 3;
    d.cfg.countSign();
    ok = ok && d.cfg.n_negative == 0;
    check(ok, "SignCounter: |r| returned, sign flips on negative ratios, steps counted only on negative diagrams");
}

template <typename U>
static void step(U & up, diagram_cfg & cfg, simplemc::xoshiro256ss & rng){
    std::uniform_real_distribution<double> u {0., 1.};
    const double p {up.attempt()};
    if (p < 0.) { up.reject(); }
    else if (p >= u(rng)) { up.accept(); }
    else { up.reject(); }
    cfg.countSign();
}

static void chain(test::Checks & check){
    test::Diagram d {777ULL, {0.11, -0.07, 0.05}, 24, 8, {PhononMode{0.5, 2.0}, PhononMode{0.8, 3.0}}};
    diagram_cfg & cfg {d.cfg};
    add_int_ph_update addi {&cfg, &d.rng};  rm_int_ph_update rmi {&cfg, &d.rng};
    add_ext_ph_update adde {&cfg, &d.rng};  rm_ext_ph_update rme {&cfg, &d.rng};
    swp_ph_update swp {&cfg, &d.rng};       mv_tau_update mv {&cfg, &d.rng};
    chg_tau_update ch {&cfg, &d.rng};       chg_ph_momentum chw {&cfg, &d.rng};
    chg_ph_energy che {&cfg, &d.rng};
    std::uniform_int_distribution<int> pick {0, 8};

    const long STEPS {2000000};
    long mismatches {0}, dirty {0}, flips {0};
    int previous {cfg.current_sign};
    for (long s {0}; s < STEPS; ++s) {
        switch (pick(d.rng)) {
            case 0: step(addi, cfg, d.rng); break;  case 1: step(rmi, cfg, d.rng); break;
            case 2: step(adde, cfg, d.rng); break;  case 3: step(rme, cfg, d.rng); break;
            case 4: step(swp, cfg, d.rng); break;   case 5: step(mv, cfg, d.rng); break;
            case 6: step(ch, cfg, d.rng); break;    case 7: step(chw, cfg, d.rng); break;
            default: step(che, cfg, d.rng); break;
        }
        const double trace {cfg.diagram_head->right_component.trace()};
        if (cfg.current_sign != (trace < 0. ? -1 : 1)) { ++mismatches; }
        if (cfg.current_sign != previous) { ++flips; previous = cfg.current_sign; }
        if (s % 1000 == 0 && !numerical::sanitizeDiagram(&cfg).clean()) { ++dirty; }
    }

    const std::uint64_t local {addi.sign.n_negative + rmi.sign.n_negative + adde.sign.n_negative + rme.sign.n_negative
                             + swp.sign.n_negative + mv.sign.n_negative + ch.sign.n_negative + chw.sign.n_negative
                             + che.sign.n_negative};
    check(mismatches == 0 && dirty == 0,
          "tracked sign == sign(tr P) after each of %ld steps (%ld sign changes), sanitizer clean every 1000 steps",
          STEPS, flips);
    check(cfg.n_negative == local,
          "global negative count %llu == sum of the updates' counts %llu (%.4f%% of the steps negative)",
          static_cast<unsigned long long>(cfg.n_negative), static_cast<unsigned long long>(local),
          100. * static_cast<double>(cfg.n_negative) / STEPS);
}

int main(){
    test::Checks check {"sign tracking (|ratio| sampling, current sign, negative counters)"};
    test::setLK(test::AlAs);
    unit(check);
    chain(check);
    return check.exit_code();
}
