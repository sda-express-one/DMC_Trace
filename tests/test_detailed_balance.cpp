// Detailed balance of mv_tau and chg_tau: chains running ONE update on a hand-built diagram, whose
// stationary averages must match the exact averages of the target weight, obtained by numerical
// integration. The topology is fixed (neither update adds or removes lines), so the target is a known
// function of the free times only:
//
//   mv_tau, one line head -> v1 -> v2 -> tail, tail fixed at L:
//     W(t1, t2) = tr( A_head(t1) wf1 A_mid(t2 - t1) wf2 A_end(L - t2) ) * phonon(t1, t2)
//     internal line: phonon = e^{-omega (t2 - t1)};  external line: e^{-omega (L - t2 + t1)} (wrapped).
//     The external case checks the analytic cancellation of the proposal's phonon part against the
//     wrapped propagator, for both external vertex types.
//   chg_tau, tail free, last vertex V fixed: W(d) = tr( wf(V) A_last(d) L(V) ) e^{(mu - omega_ext) d},
//     d = tau_D - tau_V, on a bare propagator and on a diagram whose last segment is energetic and long
//     after the start (E_min tau_D ~ 1000: the regime where the old ratio underflowed to 0/0).
//
// Errors from the spread over independent chains, each started at a random point after a burn-in.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>
#include "test_common.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"

using M3 = Eigen::Matrix3d;

static M3 action(const std::array<double, 3> & e, double d){
    return M3(Eigen::Vector3d(std::exp(-e[0]*d), std::exp(-e[1]*d), std::exp(-e[2]*d)).asDiagonal());
}

struct MeanErr { double mean, err; };
static MeanErr spread(const std::vector<double> & x){
    double m {0.}; for (double v : x) { m += v; } m /= static_cast<double>(x.size());
    double s {0.}; for (double v : x) { s += (v - m)*(v - m); }
    return {m, std::sqrt(s / static_cast<double>(x.size() - 1) / static_cast<double>(x.size()))};
}

// head -> first -> second -> tail. Internal line: first = creation (+1), second = annihilation (-1),
// momentum p - w between them. External line (case 1): first = annihilation (-2), second = creation (+2),
// momentum p - w OUTSIDE them, i.e. on the head and last segments.
static void build_line(test::Diagram & d, bool external, std::array<double, 3> p, std::array<double, 3> w,
                       double t1, double t2, double L){
    diagram_cfg & cfg {d.cfg};
    Vertex * h {cfg.diagram_head}; Vertex * t {cfg.diagram_tail};
    Vertex * a {cfg.drawVertexFromPool()}; Vertex * b {cfg.drawVertexFromPool()};
    cfg.addVertex(a, h); cfg.addVertex(b, a);
    cfg.current_tau_length = L; t->tau = L;
    h->tau = 0.; h->tau_next = t1; a->tau = t1; a->tau_next = t2; b->tau = t2; b->tau_next = L;
    const std::array<double, 3> shifted {p[0] - w[0], p[1] - w[1], p[2] - w[2]};
    a->w = w; b->w = w;
    a->ph_energy = b->ph_energy = d.modes.phonon_mode_pool[0].phonon_energy;
    a->diel_response = b->diel_response = d.modes.phonon_mode_pool[0].diel_response;
    a->conj_vertex = b; b->conj_vertex = a;
    if (!external) {
        a->type = +1; b->type = -1; h->k = p; a->k = shifted; b->k = p;
        d.internal.addVertexPointers(a, b);
    } else {
        a->type = -2; b->type = +2; h->k = shifted; a->k = p; b->k = shifted;
        d.external.addVertexPointers(b, a);
    }
    numerical::sanitizeDiagram(&cfg);
}

static void mv_case(test::Checks & check, const char * label, bool external, std::array<double, 3> w){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    const double L {2.0}, omega {0.5};
    const int CHAINS {12};
    const long BURN {200000}, STEPS {2000000};

    // exact averages over 0 < t1 < t2 < L, with s = t2 - t1 = (L - t1) u^4 to resolve a narrow peak at s ~ 0
    double exact_t1 {0.}, exact_s {0.}, exact_t2 {0.};
    {
        test::Diagram d {1ULL, p, 4, 4};
        build_line(d, external, p, w, 0.6, 1.4, L);
        const Vertex * h {d.cfg.diagram_head}; const Vertex * a {h->next}; const Vertex * b {a->next};
        const std::array<double, 3> Eh {h->electronEnergy()}, Ea {a->electronEnergy()}, Eb {b->electronEnergy()};
        const M3 wf_a {a->vertex_wf_component}, wf_b {b->vertex_wf_component};
        double Z {0.};
        const int N1 {1500}, NU {3000};
        for (int i {0}; i < N1; ++i) {
            const double t1 {(i + 0.5) * L / N1};
            for (int j {0}; j < NU; ++j) {
                const double uu {(j + 0.5) / NU}, s {(L - t1) * std::pow(uu, 4)};
                const double jac {(L - t1) * 4. * std::pow(uu, 3) / NU * (L / N1)};
                const double t2 {t1 + s};
                const double ph {external ? std::exp(-omega*(L - t2 + t1)) : std::exp(-omega*s)};
                const double W {(action(Eh, t1) * wf_a * action(Ea, s) * wf_b * action(Eb, L - t2)).trace() * ph};
                Z += W*jac; exact_t1 += t1*W*jac; exact_s += s*W*jac; exact_t2 += t2*W*jac;
            }
        }
        exact_t1 /= Z; exact_s /= Z; exact_t2 /= Z;
    }

    std::vector<double> t1s, ss, t2s;
    long accepted {0};
    for (int c {0}; c < CHAINS; ++c) {
        std::mt19937 start {static_cast<unsigned>(100 + c)};
        std::uniform_real_distribution<double> pos {0.05, 1.9};
        const double t1 {pos(start)};
        test::Diagram d {500ULL + static_cast<unsigned long long>(c), p, 4, 4};
        build_line(d, external, p, w, t1, std::min(t1 + 0.05, L - 0.01), L);
        mv_tau_update mv {&d.cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        const Vertex * a {d.cfg.diagram_head->next}; const Vertex * b {a->next};
        double s1 {0.}, ss_ {0.}, s2 {0.};
        for (long s {-BURN}; s < STEPS; ++s) {
            const double r {mv.attempt()};
            if (r > 0. && u(d.rng) < r) { mv.accept(); if (s >= 0) { ++accepted; } }
            if (s >= 0) { s1 += a->tau; ss_ += b->tau - a->tau; s2 += b->tau; }
        }
        t1s.push_back(s1 / STEPS); ss.push_back(ss_ / STEPS); t2s.push_back(s2 / STEPS);
    }
    const MeanErr m1 {spread(t1s)}, ms {spread(ss)}, m2 {spread(t2s)};
    const double p1 {(m1.mean - exact_t1)/m1.err}, ps {(ms.mean - exact_s)/ms.err}, p2 {(m2.mean - exact_t2)/m2.err};
    check(std::abs(p1) < 4.5 && std::abs(ps) < 4.5 && std::abs(p2) < 4.5,
          "mv_tau, %s: <t1> %.5f vs %.5f (%+.1f sig)  <t2-t1> %.5f vs %.5f (%+.1f sig)  <t2> %.5f vs %.5f (%+.1f sig), acceptance %.2f",
          label, m1.mean, exact_t1, p1, ms.mean, exact_s, ps, m2.mean, exact_t2, p2,
          static_cast<double>(accepted) / (CHAINS * STEPS));
}

static void chg_case(test::Checks & check, const char * label, bool with_line){
    const int CHAINS {12};
    const long BURN {50000}, STEPS {1000000};
    const double omega {0.5};
    const std::array<double, 3> p_bare {0.4, -0.3, 0.2};
    const std::array<double, 3> p {0.11, -0.07, 0.05}, end_k {7.5, -5.0, 6.25};
    const std::array<double, 3> w {p[0] - end_k[0], p[1] - end_k[1], p[2] - end_k[2]};   // p - w = end_k at both ends
    const double tau_V {30.0};

    // Diagram is neither copyable nor movable, hence the unique_ptr
    auto make = [&](unsigned long long seed){
        auto d {std::make_unique<test::Diagram>(seed, with_line ? p : p_bare, 4, 4, 50.0)};
        if (with_line) { build_line(*d, true, p, w, 0.5, tau_V, tau_V + 0.01); }
        return d;
    };

    double exact {0.};
    {
        const auto d {make(1ULL)};
        const Vertex * V {d->cfg.diagram_tail->prev};
        const std::array<double, 3> e {V->electronEnergy()};
        const double dmax {d->cfg.tau_max - V->tau}, rate {d->cfg.chem_pot - (with_line ? omega : 0.)};
        double Z {0.};
        const int NG {400000};
        for (int i {0}; i < NG; ++i) {
            const double uu {(i + 0.5) / NG}, x {dmax * uu * uu}, jac {dmax * 2. * uu / NG};   // finer near d = 0
            const double W {(V->vertex_wf_component * action(e, x) * V->left_component).trace() * std::exp(rate * x)};
            Z += W*jac; exact += x*W*jac;
        }
        exact /= Z;
    }

    std::vector<double> ds;
    for (int c {0}; c < CHAINS; ++c) {
        const auto d {make(800ULL + static_cast<unsigned long long>(c))};
        chg_tau_update ch {&d->cfg, &d->rng};
        std::uniform_real_distribution<double> u {0., 1.};
        const double tV {d->cfg.diagram_tail->prev->tau};
        double acc {0.};
        for (long s {-BURN}; s < STEPS; ++s) {
            const double r {ch.attempt()};
            if (r > 0. && u(d->rng) < r) { ch.accept(); }
            if (s >= 0) { acc += d->cfg.diagram_tail->tau - tV; }
        }
        ds.push_back(acc / STEPS);
    }
    const MeanErr m {spread(ds)};
    const double pull {(m.mean - exact) / m.err};
    check(std::abs(pull) < 4.5, "chg_tau, %s: <tau_D - tau_V> %.6f +- %.6f vs exact %.6f (%+.1f sig)", label, m.mean, m.err, exact, pull);
}

int main(){
    test::Checks check {"detailed balance of mv_tau and chg_tau against exact averages"};
    test::setLK(test::AlAs);
    mv_case(check, "internal line, moderate w", false, {0.6, -0.4, 0.5});
    mv_case(check, "internal line, large w   ", false, {6.0, -4.0, 5.0});
    mv_case(check, "external line            ", true,  {0.6, -0.4, 0.5});
    chg_case(check, "bare propagator (mu cancellation)", false);
    chg_case(check, "external line, E_min tau_D ~ 1000", true);
    return check.exit_code();
}
