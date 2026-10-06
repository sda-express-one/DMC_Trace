// Detailed balance of mv_tau, chg_tau and chg_ph_momentum: chains running ONE update on a hand-built diagram, whose
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
//   chg_ph_momentum, one line at fixed times, for each of the three line kinds: pi(w) ~ T(w) |g(w)|^2,
//     averages of |w|^2, |w| and w.p^ against 3D quadrature. Plus the returned ratio itself on general
//     multi-line diagrams from a chain with all updates, against the diagram rebuilt from scratch.
//
// Errors from the spread over independent chains, each started at a random point after a burn-in.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <random>
#include <vector>
#include "test_common.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"

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

// head -> first -> second -> tail, with momentum p carried by the segment the line does not touch:
//   internal:            first = creation (+1), second = annihilation (-1); p - w between them.
//   external, ann first: first = annihilation (-2), second = creation (+2) (case 1); p - w on the head
//                        and last segments, p between them.
//   external, cre first: first = creation (+2), second = annihilation (-2) (case 2); p - w on the head
//                        and last segments, p - 2w between them (the line wraps over it twice).
enum class Line { internal, ext_ann_first, ext_cre_first };

static std::array<std::array<double, 3>, 3> line_momenta(Line kind, std::array<double, 3> p, std::array<double, 3> w){
    auto shifted = [&](double c){ return std::array<double, 3>{p[0] - c*w[0], p[1] - c*w[1], p[2] - c*w[2]}; };
    switch (kind) {
        case Line::internal:      return {shifted(0.), shifted(1.), shifted(0.)};
        case Line::ext_ann_first: return {shifted(1.), shifted(0.), shifted(1.)};
        default:                  return {shifted(1.), shifted(2.), shifted(1.)};
    }
}

static void build_line(test::Diagram & d, Line kind, std::array<double, 3> p, std::array<double, 3> w,
                       double t1, double t2, double L){
    diagram_cfg & cfg {d.cfg};
    Vertex * h {cfg.diagram_head}; Vertex * t {cfg.diagram_tail};
    Vertex * a {cfg.drawVertexFromPool()}; Vertex * b {cfg.drawVertexFromPool()};
    cfg.addVertex(a, h); cfg.addVertex(b, a);
    cfg.current_tau_length = L; t->tau = L;
    h->tau = 0.; h->tau_next = t1; a->tau = t1; a->tau_next = t2; b->tau = t2; b->tau_next = L;
    const std::array<std::array<double, 3>, 3> k {line_momenta(kind, p, w)};
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
        build_line(d, external ? Line::ext_ann_first : Line::internal, p, w, 0.6, 1.4, L);
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
        build_line(d, external ? Line::ext_ann_first : Line::internal, p, w, t1, std::min(t1 + 0.05, L - 0.01), L);
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
        if (with_line) { build_line(*d, Line::ext_ann_first, p, w, 0.5, tau_V, tau_V + 0.01); }
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

// Gauss-Legendre nodes and weights on [a, b]
static void gauss_legendre(int n, double a, double b, std::vector<double> & x, std::vector<double> & wt){
    x.resize(n); wt.resize(n);
    for (int i {0}; i < n; ++i) {
        double z {std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5))}, dp {0.};
        for (int it {0}; it < 100; ++it) {
            double p1 {1.}, p2 {0.};
            for (int j {1}; j <= n; ++j) { const double p3 {p2}; p2 = p1; p1 = ((2.*j - 1.)*z*p2 - (j - 1.)*p3) / j; }
            dp = n * (z*p1 - p2) / (z*z - 1.);
            const double dz {p1 / dp};
            z -= dz;
            if (std::abs(dz) < 1e-15) { break; }
        }
        x[i] = 0.5*(a + b) - 0.5*(b - a)*z;
        wt[i] = (b - a) / ((1. - z*z) * dp*dp);
    }
}

// electronic trace of the one-line diagram at phonon momentum w, rebuilt from scratch
static double line_trace(Line kind, std::array<double, 3> p, std::array<double, 3> w, const std::array<double, 3> & dur){
    const std::array<std::array<double, 3>, 3> ks {line_momenta(kind, p, w)};
    M3 prod {M3::Identity()}, U_prev {M3::Identity()};
    for (int s {0}; s < 3; ++s) {
        const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(ks[s])};
        const M3 U {eig.block<3,3>(1,0)};
        const std::array<double, 3> m {weight::LKMatrix::computeEffMassfromEigenval(std::array<double, 3>{eig(0,0), eig(0,1), eig(0,2)})};
        const double k2 {ks[s][0]*ks[s][0] + ks[s][1]*ks[s][1] + ks[s][2]*ks[s][2]};
        if (s > 0) { prod = prod * Coupling::LKOverlap::computeMatrix(U_prev, U); }
        prod = prod * action({k2/(2.*m[0]), k2/(2.*m[1]), k2/(2.*m[2])}, dur[s]);
        U_prev = U;
    }
    return prod.trace();
}

// chg_ph_momentum on a single line with fixed times. The phonon propagator does not depend on w, so
// the target is pi(w) d^3w ~ T(w) |g(w)|^2 d^3w ~ T(w) dr dOmega: the 1/|w|^2 of the coupling cancels
// the r^2 of the measure, and the exact averages are smooth integrals in spherical coordinates
// (Gauss-Legendre in r and cos(theta), trapezoid in phi). Updates reject r <= 0, so the chain samples
// T restricted to T > 0, and so does the integral.
static void chg_w_case(test::Checks & check, const char * label, Line kind){
    const std::array<double, 3> p {0.30, -0.20, 0.25};
    const double L {2.0}, t1 {0.6}, t2 {1.4};
    const std::array<double, 3> dur {t1, t2 - t1, L - t2};
    const double pn {std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])};
    const std::array<double, 3> e {p[0]/pn, p[1]/pn, p[2]/pn};
    const int CHAINS {12};
    const long BURN {50000}, STEPS {800000};

    // width of T in |w|: the smallest LK eigenvalue over directions and the shortest time w enters for
    std::vector<double> xc, wc, xr, wr;
    gauss_legendre(48, -1., 1., xc, wc);
    const int NPHI {96};
    double lam_min {1e300};
    for (double c : xc) {
        for (int j {0}; j < NPHI; ++j) {
            const double ph {2.*std::numbers::pi*j/NPHI}, sn {std::sqrt(1. - c*c)};
            const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian({sn*std::cos(ph), sn*std::sin(ph), c})};
            lam_min = std::min({lam_min, eig(0,0), eig(0,1), eig(0,2)});
        }
    }
    const double D {kind == Line::internal ? t2 - t1 : L - (t2 - t1)};
    const double R {pn + std::sqrt(45. / (lam_min * D))};
    gauss_legendre(160, 0., R, xr, wr);

    double Z {0.}, ex_r2 {0.}, ex_r {0.}, ex_proj {0.}, Z_neg {0.};
    for (std::size_t ir {0}; ir < xr.size(); ++ir) {
        for (std::size_t ic {0}; ic < xc.size(); ++ic) {
            const double sn {std::sqrt(1. - xc[ic]*xc[ic])};
            for (int j {0}; j < NPHI; ++j) {
                const double ph {2.*std::numbers::pi*j/NPHI}, r {xr[ir]};
                const std::array<double, 3> w {r*sn*std::cos(ph), r*sn*std::sin(ph), r*xc[ic]};
                const double T {line_trace(kind, p, w, dur)}, jac {wr[ir]*wc[ic]*2.*std::numbers::pi/NPHI};
                if (T <= 0.) { Z_neg -= T*jac; continue; }
                Z += T*jac; ex_r2 += r*r*T*jac; ex_r += r*T*jac; ex_proj += (w[0]*e[0] + w[1]*e[1] + w[2]*e[2])*T*jac;
            }
        }
    }
    ex_r2 /= Z; ex_r /= Z; ex_proj /= Z;

    std::vector<double> r2s, rs, projs;
    long accepted {0};
    double worst_start {0.};
    for (int c {0}; c < CHAINS; ++c) {
        std::mt19937 start {static_cast<unsigned>(300 + c)};
        std::normal_distribution<double> g {0., 0.5};
        const std::array<double, 3> w0 {g(start), g(start), g(start)};
        test::Diagram d {900ULL + static_cast<unsigned long long>(c), p, 4, 4};
        build_line(d, kind, p, w0, t1, t2, L);
        // the hand-built diagram must be the one line_trace describes
        const double T0 {line_trace(kind, p, w0, dur)};
        worst_start = std::max(worst_start, std::abs(d.cfg.diagram_head->right_component.trace() - T0) / std::abs(T0));
        chg_ph_momentum chw {&d.cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        const Vertex * a {d.cfg.diagram_head->next};
        double s_r2 {0.}, s_r {0.}, s_proj {0.};
        for (long s {-BURN}; s < STEPS; ++s) {
            const double r {chw.attempt()};
            if (r > 0. && u(d.rng) < r) { chw.accept(); if (s >= 0) { ++accepted; } }
            if (s >= 0) {
                const std::array<double, 3> & w {a->w};
                const double r2 {w[0]*w[0] + w[1]*w[1] + w[2]*w[2]};
                s_r2 += r2; s_r += std::sqrt(r2); s_proj += w[0]*e[0] + w[1]*e[1] + w[2]*e[2];
            }
        }
        r2s.push_back(s_r2 / STEPS); rs.push_back(s_r / STEPS); projs.push_back(s_proj / STEPS);
    }
    const MeanErr m2 {spread(r2s)}, m1 {spread(rs)}, mp {spread(projs)};
    const double p2 {(m2.mean - ex_r2)/m2.err}, p1 {(m1.mean - ex_r)/m1.err}, pp {(mp.mean - ex_proj)/mp.err};
    check(worst_start < 1e-12 && std::abs(p2) < 4.5 && std::abs(p1) < 4.5 && std::abs(pp) < 4.5,
          "chg_ph_momentum, %s: <|w|^2> %.5f vs %.5f (%+.1f sig)  <|w|> %.5f vs %.5f (%+.1f sig)  <w.p^> %.5f vs %.5f (%+.1f sig), "
          "acceptance %.2f, negative-T weight %.1e",
          label, m2.mean, ex_r2, p2, m1.mean, ex_r, p1, mp.mean, ex_proj, pp,
          static_cast<double>(accepted) / (CHAINS * STEPS), Z_neg / Z);
}

// The returned ratio itself, on general diagrams (several lines, internal ones spanning other vertices,
// external lines of both orientations): a chain with all updates, and for every accepted
// chg_ph_momentum move the ratio is compared with the detailed-balance expression rebuilt from scratch,
//   r = T(y) |g(w')|^2 q(w) / (T(x) |g(w)|^2 q(w')) = T(y)/T(x) * exp(-(|w|^2 - |w'|^2) l/2),
// with the half-normal |w| / uniform direction proposal q(w) = h(|w|)/(4 pi |w|^2), whose 1/|w|^2
// cancels the coupling's. T from the sanitizer's full rebuild, l the line length the proposal uses
// (tau2 - tau1 internal, L - tau_cre + tau_ann external). Checks attempt() and accept() describe the
// same diagram. attempt() returns |r| and records the sign of r: both are compared, and the move's
// sign must carry the tracked diagram sign (cfg->current_sign) to the sign of the rebuilt trace.
static void chg_w_ratio_case(test::Checks & check){
    test::Diagram d {4242ULL, {0.11, -0.07, 0.05}, 24, 8};
    diagram_cfg * cfg {&d.cfg};
    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 7};

    const char * name[3] {"internal", "external, ann first", "external, cre first"};
    long checked[3] {}, spanning {0}, sign_errors {0}, negative_moves {0};
    double worst[3] {};
    long dirty {0};

    for (long step {0}; step < 1000000; ++step) {
        const int which {pick(d.rng)};
        if (which < 7) {
            double r {-1.};
            switch (which) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                             case 3: r = rme.attempt(); break; case 4: r = swp.attempt(); break; case 5: r = mv.attempt(); break;
                             default: r = ch.attempt(); }
            if (!(r > 0. && u(d.rng) < r)) { continue; }
            switch (which) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                             case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                             default: ch.accept(); }
            if (!numerical::sanitizeDiagram(cfg).clean()) { ++dirty; }
            continue;
        }

        const double T_before {cfg->diagram_head->right_component.trace()};
        const int sign_before {cfg->current_sign};
        const double r {chw.attempt()};
        if (!(r > 0. && u(d.rng) < r)) { continue; }

        const std::array<double, 3> w_old {chw.ptr_one->w}, w_new {chw.w_proposed};
        const int b {static_cast<int>(chw.branch)};
        const double l {b == 0 ? chw.ptr_two->tau - chw.ptr_one->tau
                               : cfg->diagram_tail->tau - chw.ptr_two->tau + chw.ptr_one->tau};
        if (b == 0 && chw.ptr_one->next != chw.ptr_two) { ++spanning; }

        chw.accept();
        const int sign_tracked {cfg->current_sign};      // before the sanitizer re-derives it
        const numerical::SanitizeReport rep {numerical::sanitizeDiagram(cfg)};
        if (!rep.clean()) { ++dirty; }

        const double n_old {w_old[0]*w_old[0] + w_old[1]*w_old[1] + w_old[2]*w_old[2]};
        const double n_new {w_new[0]*w_new[0] + w_new[1]*w_new[1] + w_new[2]*w_new[2]};
        const double expected {rep.trace_after / T_before * std::exp(-(n_old - n_new) * l / 2.)};
        worst[b] = std::max(worst[b], std::abs(r / std::abs(expected) - 1.));
        const int expected_sign {expected < 0. ? -1 : 1};
        if (chw.sign.proposed_sign != expected_sign || sign_tracked != sign_before * expected_sign
            || sign_tracked != (rep.trace_after < 0. ? -1 : 1)) { ++sign_errors; }
        if (expected_sign < 0) { ++negative_moves; }
        ++checked[b];
    }

    for (int b {0}; b < 3; ++b) {
        check(checked[b] >= 100 && worst[b] < 1e-9 && dirty == 0,
              "chg_ph_momentum ratio vs rebuilt diagram, %-19s: %5ld accepted moves, worst relative deviation %.1e",
              name[b], checked[b], worst[b]);
    }
    check(spanning >= 100, "internal moves on lines spanning other vertices: %ld", spanning);
    check(sign_errors == 0, "sign of the ratio and tracked diagram sign vs rebuilt trace: %ld errors over %ld sign-flipping moves",
          sign_errors, negative_moves);
}

int main(){
    test::Checks check {"detailed balance of mv_tau, chg_tau and chg_ph_momentum"};
    test::setLK(test::AlAs);
    mv_case(check, "internal line, moderate w", false, {0.6, -0.4, 0.5});
    mv_case(check, "internal line, large w   ", false, {6.0, -4.0, 5.0});
    mv_case(check, "external line            ", true,  {0.6, -0.4, 0.5});
    chg_case(check, "bare propagator (mu cancellation)", false);
    chg_case(check, "external line, E_min tau_D ~ 1000", true);
    chg_w_case(check, "internal line      ", Line::internal);
    chg_w_case(check, "external, ann first", Line::ext_ann_first);
    chg_w_case(check, "external, cre first", Line::ext_cre_first);
    chg_w_ratio_case(check);
    return check.exit_code();
}
