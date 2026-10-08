// New vertex times must never tie with an existing one. At tau ~ 1e5 a double resolves times only to
// ~1.5e-11 (one ulp), so a proposed time can round exactly onto a neighbouring vertex, the head or the
// tail, which would create a zero-length segment. Exact ties have probability zero in exact arithmetic
// and are unreachable by any move, so they must be rejected - with exact comparisons, no tolerance.
// The regime is forced by geometry, not by luck:
//   - internal: a line whose two vertices are ONE ulp apart near tau = 1e5; a new vertex drawn inside
//     that segment, tau_init + u * ulp, always rounds onto one of its ends;
//   - external: a phonon energy so large that tau_two = tau_D - Exp(omega) rounds onto the tail
//     (tau_D = 1e5) for a sizeable fraction of the draws.
// Every proposal whose times are not strictly inside their segments must return -1, and a chain with
// all updates started from the internal configuration must never produce a broken time ordering.
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
#include "updates/str_diagram.hpp"
#include "updates/scl_diagram.hpp"

// is tau strictly inside some segment of the diagram (no tie with any vertex, the head or the tail)?
static bool strictly_placed(const diagram_cfg & cfg, double tau){
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        if (v->tau < tau && tau < v->tau_next) { return true; }
    }
    return false;
}

static void internal_case(test::Checks & check){
    const std::array<double, 3> p {0.11, -0.07, 0.05}, w {0.6, -0.4, 0.5};
    const double ta {1e5};
    const double tb {std::nextafter(ta, 2e5)};     // one ulp later
    const double L {2e5};
    test::Diagram d {4321ULL, p, 24, 8, 3e5, -1.};
    diagram_cfg * cfg {&d.cfg};
    {
        Vertex * h {cfg->diagram_head}; Vertex * t {cfg->diagram_tail};
        Vertex * a {cfg->drawVertexFromPool()}; Vertex * b {cfg->drawVertexFromPool()};
        cfg->addVertex(a, h); cfg->addVertex(b, a);
        cfg->current_tau_length = L; t->tau = L;
        h->tau = 0.; h->tau_next = ta; a->tau = ta; a->tau_next = tb; b->tau = tb; b->tau_next = L;
        h->k = p; a->k = {p[0] - w[0], p[1] - w[1], p[2] - w[2]}; b->k = p;
        a->w = w; b->w = w;
        a->ph_energy = b->ph_energy = 0.5; a->diel_response = b->diel_response = 2.0;
        a->conj_vertex = b; b->conj_vertex = a;
        a->type = +1; b->type = -1;
        d.internal.addVertexPointers(a, b);
        a->vertexStrength(); b->vertexStrength();
        numerical::sanitizeDiagram(cfg);
    }
    check(tb > ta && tb - ta < 2e-11 && numerical::sanitizeDiagram(cfg).broken_links == 0,
          "internal start: a line one ulp long (%.2e) at tau = 1e5, a valid diagram", tb - ta);

    // 1. add_int_ph attempts on it, nothing accepted
    add_int_ph_update addi {cfg, &d.rng};
    long ties {0}, ties_passed {0}, genuine {0};
    for (int n {0}; n < 300000; ++n) {
        const double r {addi.attempt()};
        const bool tie_one {!strictly_placed(*cfg, addi.tau_one)};
        const bool tie_two {!strictly_placed(*cfg, addi.tau_two) || !(addi.tau_one < addi.tau_two)};
        if (tie_one) { ++ties; }
        if (r != -1.) {
            ++genuine;
            if (tie_one || tie_two) { ++ties_passed; }
        }
    }
    check(ties > 1000 && ties_passed == 0,
          "add_int_ph: %ld proposals with tau_one on an existing vertex, %ld of the %ld non-rejected proposals tie (must be 0)",
          ties, ties_passed, genuine);

    // 2. a chain with all updates from there: the time ordering must never break
    rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
    str_diagram_update strd {cfg, &d.rng}; scl_diagram_update scld {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 9};
    long broken {0}, accepted {0};
    for (long s {0}; s < 100000; ++s) {
        const int which {pick(d.rng)};
        double r {-1.};
        switch (which) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                         case 3: r = rme.attempt(); break; case 4: r = swp.attempt(); break; case 5: r = mv.attempt(); break;
                         case 6: r = ch.attempt(); break; case 7: r = chw.attempt(); break; case 8: r = strd.attempt(); break;
                         default: r = scld.attempt(); }
        if (!(r >= 0. && u(d.rng) < r)) { continue; }
        switch (which) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                         case 6: ch.accept(); break; case 7: chw.accept(); break; case 8: strd.accept(); break;
                         default: scld.accept(); }
        ++accepted;
        if (numerical::sanitizeDiagram(cfg).broken_links != 0) { ++broken; break; }
    }
    check(broken == 0, "chain with all updates from the one-ulp line: %ld accepted moves, time ordering broken: %s",
          accepted, broken == 0 ? "never" : "yes");
}

static void external_case(test::Checks & check){
    const double L {1e5};
    // omega so large that tau_two = tau_D - Exp(omega) is within half an ulp of tau_D for ~7% of the draws
    test::Diagram d {8765ULL, {0.11, -0.07, 0.05}, 4, 8, {PhononMode{1e10, 2.0}}, 2e5, -1.};
    diagram_cfg * cfg {&d.cfg};
    cfg->current_tau_length = L;
    cfg->diagram_tail->tau = L;
    cfg->diagram_head->tau_next = L;
    numerical::sanitizeDiagram(cfg);

    add_ext_ph_update adde {cfg, &d.rng};
    long ties {0}, ties_passed {0}, genuine {0};
    for (int n {0}; n < 300000; ++n) {
        const double r {adde.attempt()};
        const bool tie {!strictly_placed(*cfg, adde.tau_one) || !strictly_placed(*cfg, adde.tau_two) || !(adde.tau_one != adde.tau_two)};
        if (tie) { ++ties; }
        if (r != -1.) {
            ++genuine;
            if (tie) { ++ties_passed; }
        }
    }
    check(ties > 1000 && ties_passed == 0,
          "add_ext_ph: %ld proposals with a time on the head/tail, %ld of the %ld non-rejected proposals tie (must be 0)",
          ties, ties_passed, genuine);
}

int main(){
    test::Checks check {"new vertex times never tie (tau ~ 1e5)"};
    test::setLK(test::AlAs);
    internal_case(check);
    external_case(check);
    return check.exit_code();
}
