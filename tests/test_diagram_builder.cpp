// diagram_builder: building diagrams directly (e.g. overdispersed initial diagrams for independent chains).
//
// 1. Primitives: one-line diagrams of each kind (internal, external annihilation first, external creation
//    first) built with reset/addInternalLine/addExternalLine/assignMomenta/finalize equal the hand-built ones
//    (the layout used in test_detailed_balance): same times, types, momenta, conjugates, manager slots, trace.
// 2. Failure rules: invalid times, ties with an existing vertex and a full manager are refused and leave the
//    diagram unchanged.
// 3. Random diagrams, many seeds, for several band structures (AlAs valence C != 0, AlAs conduction C = 0,
//    isotropic), p != 0 and two phonon modes: sanitizer clean on the rebuilt diagram, momentum conserved,
//    counts within the spec (clamped at capacity), finite non-zero trace, different seeds give different
//    diagrams; reset() returns every vertex to the pool.
// 4. Usable as a start: chains with all updates started from random high-order diagrams stay clean and reach
//    the same ground-state energy as a chain started from order 0.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>
#include "test_common.hpp"
#include "diagram/diagram_builder.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "measurements/gs_energy.hpp"

enum class Line { internal, ext_ann_first, ext_cre_first };

// the hand-built one-line diagram of test_detailed_balance (head -> first -> second -> tail)
static void build_line_by_hand(test::Diagram & d, Line kind, std::array<double, 3> p, std::array<double, 3> w,
                               double t1, double t2, double L){
    diagram_cfg & cfg {d.cfg};
    auto shifted = [&](double c){ return std::array<double, 3>{p[0] - c*w[0], p[1] - c*w[1], p[2] - c*w[2]}; };
    const std::array<std::array<double, 3>, 3> k {
        kind == Line::internal ? std::array<std::array<double, 3>, 3>{shifted(0.), shifted(1.), shifted(0.)}
      : kind == Line::ext_ann_first ? std::array<std::array<double, 3>, 3>{shifted(1.), shifted(0.), shifted(1.)}
      : std::array<std::array<double, 3>, 3>{shifted(1.), shifted(2.), shifted(1.)}};
    Vertex * h {cfg.diagram_head}; Vertex * t {cfg.diagram_tail};
    Vertex * a {cfg.drawVertexFromPool()}; Vertex * b {cfg.drawVertexFromPool()};
    cfg.addVertex(a, h); cfg.addVertex(b, a);
    cfg.current_tau_length = L; t->tau = L;
    h->tau = 0.; h->tau_next = t1; a->tau = t1; a->tau_next = t2; b->tau = t2; b->tau_next = L;
    h->k = k[0]; a->k = k[1]; b->k = k[2];
    a->w = w; b->w = w;
    a->ph_energy = b->ph_energy = d.modes.phonon_mode_pool[0].phonon_energy;
    a->diel_response = b->diel_response = d.modes.phonon_mode_pool[0].diel_response;
    a->conj_vertex = b; b->conj_vertex = a;
    switch (kind) {
        case Line::internal:      a->type = +1; b->type = -1; d.internal.addVertexPointers(a, b); break;
        case Line::ext_ann_first: a->type = -2; b->type = +2; d.external.addVertexPointers(b, a); break;
        case Line::ext_cre_first: a->type = +2; b->type = -2; d.external.addVertexPointers(a, b); break;
    }
    a->vertexStrength(); b->vertexStrength();
    numerical::sanitizeDiagram(&cfg);
}

static int count_free(const diagram_cfg & cfg){
    int n {0};
    for (const Vertex * v {cfg.free_stack}; v != nullptr; v = v->next) { ++n; }
    return n;
}

static void primitives(test::Checks & check){
    const std::array<double, 3> p {0.11, -0.07, 0.05}, w {0.6, -0.4, 0.5};
    const double t1 {2.5}, t2 {7.0}, L {10.};
    const char * names[3] {"internal           ", "external, ann first", "external, cre first"};
    for (int c {0}; c < 3; ++c) {
        const Line kind {static_cast<Line>(c)};
        test::Diagram ref {1ULL, p, 24, 8};
        build_line_by_hand(ref, kind, p, w, t1, t2, L);

        test::Diagram d {1ULL, p, 24, 8};
        diagram_builder::reset(d.cfg, L);
        const bool ok {kind == Line::internal ? diagram_builder::addInternalLine(d.cfg, t1, t2, w, 0)
                     : kind == Line::ext_ann_first ? diagram_builder::addExternalLine(d.cfg, t2, t1, w, 0)
                     : diagram_builder::addExternalLine(d.cfg, t1, t2, w, 0)};
        diagram_builder::assignMomenta(d.cfg, p);
        const numerical::SanitizeReport rep {diagram_builder::finalize(d.cfg)};

        double dev {0.};
        bool same {ok};
        const Vertex * a {ref.cfg.diagram_head}; const Vertex * b {d.cfg.diagram_head};
        for (; a != nullptr && b != nullptr; a = a->next, b = b->next) {
            same = same && a->tau == b->tau && a->tau_next == b->tau_next && a->type == b->type && a->index == b->index
                        && (a->conj_vertex == nullptr) == (b->conj_vertex == nullptr)
                        && (a->conj_vertex == nullptr || a->conj_vertex->tau == b->conj_vertex->tau);
            for (int i {0}; i < 3; ++i) { dev = std::max({dev, std::abs(a->k[i] - b->k[i]), std::abs(a->w[i] - b->w[i])}); }
        }
        same = same && a == nullptr && b == nullptr;
        const double dtr {std::abs(ref.cfg.diagram_head->right_component.trace() - d.cfg.diagram_head->right_component.trace())};
        check(same && dev < 1e-15 && dtr < 1e-12 && rep.clean(),
              "%s: built = hand-built (times, types, slots, conjugates; momenta %.1e, trace %.1e), sanitizer clean", names[c], dev, dtr);
    }
}

static void failures(test::Checks & check){
    const std::array<double, 3> p {0., 0., 0.}, w {0.3, 0.2, -0.1};
    test::Diagram d {2ULL, p, 4, 2};      // room for 2 internal lines and 1 external line
    diagram_cfg & cfg {d.cfg};
    diagram_builder::reset(cfg, 10.);
    const bool first {diagram_builder::addInternalLine(cfg, 2., 5., w, 0)};
    const int free_before {count_free(cfg)};
    bool refused {true};
    refused = refused && !diagram_builder::addInternalLine(cfg, 5., 4., w, 0);      // annihilation before creation
    refused = refused && !diagram_builder::addInternalLine(cfg, 3., 3., w, 0);      // equal times
    refused = refused && !diagram_builder::addInternalLine(cfg, 0., 4., w, 0);      // at the head
    refused = refused && !diagram_builder::addInternalLine(cfg, 1., 10., w, 0);     // at the tail
    refused = refused && !diagram_builder::addInternalLine(cfg, 1., 5., w, 0);      // tie with a vertex
    refused = refused && !diagram_builder::addExternalLine(cfg, 2., 8., w, 0);      // tie with a vertex
    refused = refused && !diagram_builder::addInternalLine(cfg, 1., 3., w, 7);      // no such mode
    const bool unchanged {count_free(cfg) == free_before && cfg.internal_ph_manager->current_length == 2
                          && cfg.external_ph_manager->current_length == 0};
    const bool second {diagram_builder::addInternalLine(cfg, 1., 3., w, 0)};
    const bool full {!diagram_builder::addInternalLine(cfg, 6., 7., w, 0)};       // internal manager full
    const bool ext {diagram_builder::addExternalLine(cfg, 8., 6., w, 0)};
    const bool ext_full {!diagram_builder::addExternalLine(cfg, 9., 9.5, w, 0)};   // external manager full
    diagram_builder::assignMomenta(cfg, p);
    const bool clean {diagram_builder::finalize(cfg).clean()};
    check(first && refused && unchanged && second && full && ext && ext_full && clean,
          "invalid times, ties, unknown modes and full managers refused, diagram unchanged; valid lines accepted, clean");
}

struct BandCase { const char * name; std::array<double, 3> abc; };

static void random_diagrams(test::Checks & check){
    const BandCase cases[3] {{"AlAs valence (C != 0)", test::AlAs},
                             {"AlAs conduction (C = 0)", {1. / (2. * 0.897), 1. / (2. * 0.243), 0.}},
                             {"isotropic (A = B, C = 0)", {1., 1., 0.}}};
    const diagram_builder::RandomDiagramSpec spec {.n_internal_min = 0, .n_internal_max = 30, .n_external_min = 0,
                                                   .n_external_max = 4, .w_scale = 1.5};
    for (const BandCase & bc : cases) {
        test::setLK(bc.abc);
        test::Diagram d {77ULL, {0.2, -0.1, 0.05}, 80, 10, {PhononMode{0.5, 2.0}, PhononMode{0.8, 3.0}}, 30.};
        simplemc::xoshiro256ss rng {12345ULL};
        long built {0}, dirty {0}, bad_counts {0}, bad_trace {0};      // bad_trace: zero or non-finite
        double worst_residual {0.};
        std::set<std::pair<int, int>> orders;
        std::set<long long> first_times;
        for (int s {0}; s < 400; ++s) {
            const double tau_D {5. + 20. * (s % 7) / 6.};
            const numerical::SanitizeReport rep {diagram_builder::randomDiagram(d.cfg, rng, {0.2, -0.1, 0.05}, tau_D, spec)};
            ++built;
            if (!rep.clean()) { ++dirty; }
            worst_residual = std::max(worst_residual, rep.max_momentum_residual);
            const int n_int {d.cfg.internal_ph_manager->current_length / 2}, n_ext {d.cfg.external_ph_manager->current_length / 2};
            if (n_int > spec.n_internal_max || n_ext > spec.n_external_max || d.cfg.current_tau_length != tau_D) { ++bad_counts; }
            const double tr {d.cfg.diagram_head->right_component.trace()};
            if (!std::isfinite(tr) || tr == 0.) { ++bad_trace; }
            orders.insert({n_int, n_ext});
            first_times.insert(static_cast<long long>(d.cfg.diagram_head->tau_next * 1e9));
        }
        diagram_builder::reset(d.cfg, 10.);
        const bool reset_ok {d.cfg.diagram_head->next == d.cfg.diagram_tail && count_free(d.cfg) == d.cfg.max_vertices - 2
                             && d.cfg.internal_ph_manager->current_length == 0 && d.cfg.external_ph_manager->current_length == 0};
        check(dirty == 0 && worst_residual < 1e-12 && bad_counts == 0 && bad_trace == 0 && orders.size() > 50
              && first_times.size() > 300 && reset_ok,
              "%s: %ld random diagrams clean (%ld dirty, %ld zero/non-finite traces), momentum residual %.1e, %zu distinct (n_int, n_ext), reset frees every vertex",
              bc.name, built, dirty, bad_trace, worst_residual, orders.size());
    }

    // capacity: a spec beyond what the managers can hold is clamped
    test::setLK(test::AlAs);
    test::Diagram small {5ULL, {0., 0., 0.}, 6, 4};
    simplemc::xoshiro256ss rng {9ULL};
    const diagram_builder::RandomDiagramSpec big {.n_internal_min = 10, .n_internal_max = 20, .n_external_min = 5,
                                                  .n_external_max = 9, .w_scale = 1.};
    const numerical::SanitizeReport rep {diagram_builder::randomDiagram(small.cfg, rng, {0., 0., 0.}, 10., big)};
    check(rep.clean() && small.cfg.internal_ph_manager->current_length == 6 && small.cfg.external_ph_manager->current_length == 4,
          "spec beyond capacity clamped to the managers' limits (3 internal, 2 external lines), clean");
}

// E0 from a fixed-length chain with all length-preserving updates, started either from order 0 or from a
// random high-order diagram
static double chain_energy(bool random_start, unsigned long long seed, double & err, long & dirty){
    test::setLK({1., 1., 0.});                                    // m = 1/2; omega = 1, eps = 1: alpha = 0.5
    const std::array<double, 3> p {0., 0., 0.};
    const double L {12.};
    test::Diagram d {seed, p, 300, 16, L + 1., -1., 1., 1.};
    diagram_cfg * cfg {&d.cfg};
    simplemc::xoshiro256ss init_rng {seed + 1000ULL};
    if (random_start) {
        // far above the target order (alpha omega L = 6 lines): 40-60 internal, 0-6 external lines
        diagram_builder::randomDiagram(*cfg, init_rng, p, L, {.n_internal_min = 40, .n_internal_max = 60, .n_external_min = 0,
                                                              .n_external_max = 6, .w_scale = 3.});
    }
    else {
        diagram_builder::reset(*cfg, L);
        diagram_builder::assignMomenta(*cfg, p);
        diagram_builder::finalize(*cfg);
    }
    add_int_ph_update addi {cfg, &d.rng}; rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng}; rm_ext_ph_update rme {cfg, &d.rng};
    mv_tau_update mv {cfg, &d.rng};       swp_ph_update swp {cfg, &d.rng};
    chg_ph_momentum chw {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 6};
    gs_energy_measurement E {cfg, p, 0., true};
    // an overdispersed start relaxes in ~1.5e6 steps here (49 -> ~6 lines, large-|w| lines lingering): burn in 2e6
    const long burn_in {2000000}, measured {3000000};
    for (long s {0}; s < burn_in + measured; ++s) {
        const int w {pick(d.rng)};
        double r {-1.};
        switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                     case 3: r = rme.attempt(); break; case 4: r = mv.attempt(); break; case 5: r = swp.attempt(); break;
                     default: r = chw.attempt(); }
        if (r > 0. && u(d.rng) < r) {
            switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: mv.accept(); break; case 5: swp.accept(); break;
                         default: chw.accept(); }
        }
        if (s % 50000 == 0 && !numerical::sanitizeDiagram(cfg).clean()) { ++dirty; }
        if (s >= burn_in) { E.measure(); }
    }
    const gs_energy_measurement::result r {E.normalised()};
    err = static_cast<double>(r.energy_error);
    return static_cast<double>(r.energy);
}

static void random_starts(test::Checks & check){
    double e0 {0.}, e1 {0.};
    long dirty {0};
    const double E0 {chain_energy(false, 31ULL, e0, dirty)};
    const double E1 {chain_energy(true, 32ULL, e1, dirty)};
    const double pull {(E1 - E0) / std::hypot(e0, e1)};
    check(dirty == 0 && std::abs(pull) < 4.,
          "chains from order 0 and from a random 40-60-line start: E0 = %.4f +- %.4f vs %.4f +- %.4f (%+.1f sig), sanitizer clean throughout",
          E0, e0, E1, e1, pull);
}

int main(){
    test::Checks check {"diagram_builder"};
    test::setLK(test::AlAs);
    primitives(check);
    failures(check);
    random_diagrams(check);
    random_starts(check);
    return check.exit_code();
}
