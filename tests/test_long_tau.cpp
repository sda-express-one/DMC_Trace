// Long imaginary times: the electronic trace carries e^{-S}, S = sum_i E_min,i * duration_i (about the
// kinetic energy times tau_D), which underflows once S passes ~745 if the actions are stored in full.
// A hand-built long diagram of energetic segments (large momentum, tau_D = 120, S well above 1000):
//   - its cached trace must be a normal (non-zero, non-subnormal) number;
//   - every update's attempt() on it must return a finite ratio or the -1 sentinel;
//   - a short chain with all updates started from it must keep the cached state consistent and finite.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
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
#include "updates/chg_ph_energy.hpp"
#include "updates/str_diagram.hpp"

// S = sum over segments of E_min * duration, from the momenta and times alone
static double log_scale(const diagram_cfg & cfg){
    double S {0.};
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        const std::array<double, 3> e {v->electronEnergy()};
        S += *std::min_element(e.begin(), e.end()) * (v->tau_next - v->tau);
    }
    return S;
}

static bool normal(double x){ return std::abs(x) >= std::numeric_limits<double>::min() && std::isfinite(x); }

int main(){
    test::Checks check {"long imaginary times (trace underflow)"};
    test::setLK(test::AlAs);

    const std::array<double, 3> p {6.0, 3.0, -2.0}, w {0.3, -0.2, 0.25};
    const double t1 {20.}, t2 {80.}, L {120.};

    test::Diagram d {2468ULL, p, 24, 8, {PhononMode{0.5, 2.0}, PhononMode{0.8, 3.0}}, 200., -1.};
    diagram_cfg * cfg {&d.cfg};

    // head -> a (+1, t1) -> b (-1, t2) -> tail at L: an internal line, momentum p - w between them
    {
        Vertex * h {cfg->diagram_head}; Vertex * t {cfg->diagram_tail};
        Vertex * a {cfg->drawVertexFromPool()}; Vertex * b {cfg->drawVertexFromPool()};
        cfg->addVertex(a, h); cfg->addVertex(b, a);
        cfg->current_tau_length = L; t->tau = L;
        h->tau = 0.; h->tau_next = t1; a->tau = t1; a->tau_next = t2; b->tau = t2; b->tau_next = L;
        h->k = p; a->k = {p[0] - w[0], p[1] - w[1], p[2] - w[2]}; b->k = p;
        a->w = w; b->w = w;
        a->ph_energy = b->ph_energy = 0.5; a->diel_response = b->diel_response = 2.0;
        a->conj_vertex = b; b->conj_vertex = a;
        a->type = +1; b->type = -1;
        d.internal.addVertexPointers(a, b);
        a->vertexStrength(); b->vertexStrength();
        numerical::sanitizeDiagram(cfg);
    }
    const double S0 {log_scale(*cfg)};
    const double T0 {cfg->diagram_head->right_component.trace()};
    check(S0 > 1000. && normal(T0), "start: S = %.0f (full trace e^{-S} underflows past ~745), cached trace %.3e is a normal number", S0, T0);

    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
    chg_ph_energy che {cfg, &d.rng};       str_diagram_update strd {cfg, &d.rng};
    const char * name[10] {"add_int", "rm_int", "add_ext", "rm_ext", "swp", "mv_tau", "chg_tau", "chg_w", "chg_ph_e", "str_diag"};

    auto attempt = [&](int which){
        switch (which) { case 0: return addi.attempt(); case 1: return rmi.attempt(); case 2: return adde.attempt();
                         case 3: return rme.attempt(); case 4: return swp.attempt(); case 5: return mv.attempt();
                         case 6: return ch.attempt(); case 7: return chw.attempt(); case 8: return che.attempt();
                         default: return strd.attempt(); }
    };
    auto accept = [&](int which){
        switch (which) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                         case 6: ch.accept(); break; case 7: chw.accept(); break; case 8: che.accept(); break;
                         default: strd.accept(); }
    };

    // 1. every update attempted on the long diagram itself (nothing accepted)
    for (int which {0}; which < 10; ++which) {
        long nonfinite {0}, genuine {0};
        for (int n {0}; n < 5000; ++n) {
            const double r {attempt(which)};
            if (!std::isfinite(r)) { ++nonfinite; }
            else if (r != -1.) { ++genuine; }
        }
        check(nonfinite == 0, "%-8s on the long diagram: %ld proposals evaluated, %ld non-finite ratios", name[which], genuine, nonfinite);
    }

    // 2. a short chain with all updates from there
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 9};
    long nonfinite {0}, bad_trace {0}, dirty {0}, accepted {0};
    double S_max {0.};
    for (long s {0}; s < 20000; ++s) {
        const int which {pick(d.rng)};
        const double r {attempt(which)};
        if (!std::isfinite(r)) { ++nonfinite; }
        if (r >= 0. && u(d.rng) < r) { accept(which); ++accepted; }
        if (!normal(cfg->diagram_head->right_component.trace())) { ++bad_trace; }
        S_max = std::max(S_max, log_scale(*cfg));
        if (s % 100 == 0 && !numerical::sanitizeDiagram(cfg).clean()) { ++dirty; }
    }
    check(nonfinite == 0 && bad_trace == 0 && dirty == 0,
          "chain from the long diagram: %ld accepted, %ld non-finite ratios, %ld steps with a zero/subnormal trace, %ld dirty (max S %.0f)",
          accepted, nonfinite, bad_trace, dirty, S_max);
    return check.exit_code();
}
