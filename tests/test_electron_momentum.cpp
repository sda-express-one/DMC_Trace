// electron_momentum_measurement: the time-averaged electron momentum distribution per phonon sector.
//
// 1. Bounded maps: index(edge(i) +- eps) lands in bins i / i - 1, centers inside their bins, far tails in the outer
//    bins, the outer edges infinite.
// 2. Sectors, on a chain with all updates at a generic p: the sector of every segment (segmentSectors, the running
//    count from N_ext) equals a brute-force count of the phonons present at the segment's midpoint (an external
//    line is two pieces, [0, t_ann) and [t_cre, tau_D): both present where they overlap); the walk returns to N_ext.
// 3. Bookkeeping: the test keeps its own sums of s dtau/tau delta_{N_i,N} (times 1, k_a, k_a k_b, the radial and 3D
//    bin indicators) over the samples in full batches, with every third band matrix negated to exercise the sign;
//    the estimator's means must be their ratios to sum s. Exact identities: sum_N Z_N = 1, the radial bins of sector N
//    sum to Z_N (N >= 1), the 3D cells to 1 - Z_0.
// 4. Checkpoint round trip: identical results and k_s after save/load into an estimator built with another k_s.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <simplemc/mc/json.hpp>
#include "test_common.hpp"
#include "io/checkpoint.hpp"
#include "updates/add_internal_ph.hpp"
#include "updates/rm_internal_ph.hpp"
#include "updates/add_external_ph.hpp"
#include "updates/rm_external_ph.hpp"
#include "updates/swp_ph.hpp"
#include "updates/mv_vertex.hpp"
#include "updates/chg_tau.hpp"
#include "updates/chg_ph_momentum.hpp"
#include "measurements/electron_momentum.hpp"

namespace fs = std::filesystem;
using EM = electron_momentum_measurement;

struct Chain {
    test::Diagram d;
    add_int_ph_update addi;  rm_int_ph_update rmi;
    add_ext_ph_update adde;  rm_ext_ph_update rme;
    swp_ph_update swp;       mv_tau_update mv;
    chg_tau_update ch;       chg_ph_momentum chw;
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 7};

    Chain(unsigned long long seed, std::array<double, 3> p)
        : d(seed, p, 24, 8),
          addi(&d.cfg, &d.rng), rmi(&d.cfg, &d.rng), adde(&d.cfg, &d.rng), rme(&d.cfg, &d.rng),
          swp(&d.cfg, &d.rng), mv(&d.cfg, &d.rng), ch(&d.cfg, &d.rng), chw(&d.cfg, &d.rng) {}

    void step(){
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
    }
};

static void maps(test::Checks & check){
    bool ok {true};
    for (const bool sym : {true, false}) {
        const EM::bounded_map m {0.37, 16, sym};
        for (int i {1}; i < m.n; ++i) {
            const double e {m.edge(i)}, eps {1e-9 * (1. + std::abs(e))};
            ok = ok && m.index(e + eps) == i && m.index(e - eps) == i - 1;
        }
        for (int i {0}; i < m.n; ++i) { ok = ok && m.index(m.center(i)) == i && m.edge(i) < m.center(i) && m.center(i) < m.edge(i + 1); }
        ok = ok && m.index(1e12) == m.n - 1 && std::isinf(m.edge(m.n));
        if (sym) { ok = ok && m.index(-1e12) == 0 && std::isinf(m.edge(0)) && m.edge(0) < 0. && m.edge(m.n / 2) == 0.; }
        else { ok = ok && m.edge(0) == 0. && m.index(0.) == 0; }
    }
    check(ok, "bounded maps (symmetric and radial): edges, centers and far tails land in the right bins");
}

// phonons present at time t: internal lines over [t_cre, t_ann), external lines as two pieces [0, t_ann) and [t_cre, tau_D)
static int lines_at(const diagram_cfg & cfg, double t){
    int n {0};
    for (const VertexPointerManager * m : {cfg.internal_ph_manager, cfg.external_ph_manager}) {
        const bool ext {m == cfg.external_ph_manager};
        for (int i {0}; i < m->current_length; i += 2) {
            const double t_cre {m->selectVertex(i)->tau}, t_ann {m->selectVertex(i + 1)->tau};
            // an external line: the phonon present at 0 until t_ann, and the one emitted at t_cre present until tau_D
            // (both present on [t_cre, t_ann) when t_cre < t_ann)
            if (ext) { n += (t < t_ann ? 1 : 0) + (t >= t_cre ? 1 : 0); }
            else { n += (t >= t_cre && t < t_ann) ? 1 : 0; }
        }
    }
    return n;
}

static void sectors(test::Checks & check){
    Chain c {91ULL, {0.11, -0.07, 0.05}};
    EM m {&c.d.cfg, {0.11, -0.07, 0.05}, 0., false, 16, 0.5, 4, 8, 4, 2};
    long bad {0}, checked {0}, segs {0}, max_lines {0};
    bool head_type0 {true};
    for (long step {0}; step < 200000; ++step) {
        c.step();
        if (step % 97 != 0) { continue; }
        const diagram_cfg & cfg {c.d.cfg};
        head_type0 = head_type0 && cfg.diagram_head->type == 0;
        const std::vector<int> sec {m.segmentSectors()};
        std::size_t i {0};
        for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next, ++i) {
            const int brute {lines_at(cfg, 0.5 * (v->tau + v->tau_next))};
            if (brute != sec[i]) { ++bad; }
            max_lines = std::max<long>(max_lines, brute);
            ++segs;
        }
        ++checked;
    }
    check(bad == 0 && head_type0 && max_lines >= 3,
          "segment sectors = brute-force count of the lines spanning each segment (%ld diagrams, %ld segments, up to %ld lines, %ld mismatches)",
          checked, segs, max_lines, bad);
}

static void bookkeeping(test::Checks & check){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    const long n_steps {300000};
    const int n_max {3}, nr {8}, n3 {6};
    const double ks {0.6};

    EM::result r;
    EM::result3d r3;
    std::uint64_t n_used {0}, n_used3 {0}, n_negative {0};
    {
        Chain c {4242ULL, p};
        diagram_cfg * cfg {&c.d.cfg};
        EM m {cfg, p, 0., false, 64, ks, n3, nr, n_max, 16};
        for (long step {0}; step < n_steps; ++step) {
            c.step();
            const bool flip {step % 3 == 0};
            if (flip) { cfg->diagram_head->right_component *= -1.; }
            m.measure();
            if (flip) { cfg->diagram_head->right_component *= -1.; }
        }
        r = m.normalised();
        r3 = m.normalised3d();
        n_used = m.acc.batches().size() * m.acc.batch_count();
        n_used3 = m.acc3d.batches().size() * m.acc3d.batch_count();
        n_negative = m.n_negative;
    }

    // second pass: the same chain, summed independently
    const EM::bounded_map rm {ks, nr, false}, am {ks, n3, true};
    const auto n_sec {static_cast<std::size_t>(n_max + 1)};
    std::vector<double> Z(n_sec, 0.), rad(n_sec * static_cast<std::size_t>(nr), 0.), cells(static_cast<std::size_t>(n3 * n3 * n3), 0.);
    std::vector<std::array<double, 9>> mom(n_sec, std::array<double, 9>{});
    double sum_s {0.}, sum_s3 {0.};
    {
        Chain c {4242ULL, p};
        diagram_cfg * cfg {&c.d.cfg};
        const long last {static_cast<long>(std::max(n_used, n_used3))};
        for (long step {0}; step < last; ++step) {
            c.step();
            const double tr {(step % 3 == 0 ? -1. : 1.) * cfg->diagram_head->right_component.trace()};
            const double s {tr < 0. ? -1. : 1.};
            const bool in1 {step < static_cast<long>(n_used)}, in3 {step < static_cast<long>(n_used3)};
            if (in1) { sum_s += s; }
            if (in3) { sum_s3 += s; }
            const double tau {cfg->current_tau_length};
            for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
                const int N {lines_at(*cfg, 0.5 * (v->tau + v->tau_next))};
                const auto sec {static_cast<std::size_t>(std::min(N, n_max))};
                const double w {s * (v->tau_next - v->tau) / tau};
                const auto & k {v->k};
                const double q[3] {k[0] - p[0], k[1] - p[1], k[2] - p[2]};
                if (in1) {
                    Z[sec] += w;
                    const double mm[9] {k[0], k[1], k[2], k[0]*k[0], k[1]*k[1], k[2]*k[2], k[0]*k[1], k[0]*k[2], k[1]*k[2]};
                    for (std::size_t a {0}; a < 9; ++a) { mom[sec][a] += w * mm[a]; }
                    if (N > 0) { rad[sec * static_cast<std::size_t>(nr) + static_cast<std::size_t>(rm.index(std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])))] += w; }
                }
                if (in3 && N > 0) {
                    cells[static_cast<std::size_t>((am.index(q[0]) * n3 + am.index(q[1])) * n3 + am.index(q[2]))] += w;
                }
            }
        }
    }

    double dev {0.}, total {0.}, dev_rad_sum {0.};
    for (std::size_t N {0}; N < n_sec; ++N) {
        dev = std::max(dev, std::abs(r.Z[N] - Z[N] / sum_s));
        for (std::size_t a {0}; a < 3; ++a) { dev = std::max(dev, std::abs(r.k[N][a] - mom[N][a] / sum_s)); }
        for (std::size_t a {0}; a < 6; ++a) { dev = std::max(dev, std::abs(r.kk[N][a] - mom[N][3 + a] / sum_s)); }
        double rs {0.};
        for (std::size_t j {0}; j < static_cast<std::size_t>(nr); ++j) {
            dev = std::max(dev, std::abs(r.radial[N][j] - rad[N * static_cast<std::size_t>(nr) + j] / sum_s));
            rs += r.radial[N][j];
        }
        if (N > 0) { dev_rad_sum = std::max(dev_rad_sum, std::abs(rs - r.Z[N])); }
        total += r.Z[N];
    }
    double dev3 {0.}, sum3 {0.};
    for (std::size_t c {0}; c < cells.size(); ++c) {
        dev3 = std::max(dev3, std::abs(r3.weight[c] - cells[c] / sum_s3));
        sum3 += r3.weight[c];
    }
    check(dev < 1e-12 && dev3 < 1e-12 && n_negative > 0,
          "Z_N, <k_a>, <k_a k_b>, radial and 3D weights are the ratios of the test's own sums (%llu / %llu samples, %llu negative): "
          "deviations %.1e %.1e", static_cast<unsigned long long>(n_used), static_cast<unsigned long long>(n_used3),
          static_cast<unsigned long long>(n_negative), dev, dev3);
    // the 3D accumulator has fewer samples than acc: compare its sum with its own Z_0
    double Z0_3 {0.};
    {
        // Z_0 over the 3D accumulator's samples = 1 - (sum of its cells), checked against the independent sums
        double z0 {0.};
        Chain c {4242ULL, p};
        diagram_cfg * cfg {&c.d.cfg};
        for (long step {0}; step < static_cast<long>(n_used3); ++step) {
            c.step();
            const double tr {(step % 3 == 0 ? -1. : 1.) * cfg->diagram_head->right_component.trace()};
            const double s {tr < 0. ? -1. : 1.};
            for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
                if (lines_at(*cfg, 0.5 * (v->tau + v->tau_next)) == 0) { z0 += s * (v->tau_next - v->tau) / cfg->current_tau_length; }
            }
        }
        Z0_3 = z0 / sum_s3;
    }
    check(std::abs(total - 1.) < 1e-12 && dev_rad_sum < 1e-12 && std::abs(sum3 - (1. - Z0_3)) < 1e-12,
          "sum_N Z_N = 1 (%.1e), radial bins sum to Z_N (%.1e), 3D cells sum to 1 - Z_0 (%.1e); Z_0 = %.4f, Z_1 = %.4f",
          std::abs(total - 1.), dev_rad_sum, std::abs(sum3 - (1. - Z0_3)), r.Z[0], r.Z[1]);

    bool err_ok {true};
    for (std::size_t N {0}; N < n_sec; ++N) { err_ok = err_ok && std::isfinite(r.Z_error[N]) && r.Z_error[N] > 0.; }
    for (const double e : r3.error) { err_ok = err_ok && std::isfinite(e) && e >= 0.; }
    check(err_ok, "jackknife errors finite (Z_N > 0, 3D cells >= 0)");
}

static void round_trip(test::Checks & check){
    const std::array<double, 3> p {0.05, 0., 0.};
    Chain c {77ULL, p};
    EM m {&c.d.cfg, p, 0., false, 16, 0.8, 4, 8, 3, 4};
    for (long step {0}; step < 40000; ++step) { c.step(); m.measure(); }
    const fs::path file {fs::temp_directory_path() / "dmc_test_ckpt_electron_momentum.msgpack"};
    simplemc::save_json_file(file, [&](simplemc::json_serializer & s){ s.save_at("m", m); }, checkpoint::io);
    EM fresh {&c.d.cfg, p, 0., false, 16, 1.0, 4, 8, 3, 4};      // another k_s: restored from the file
    simplemc::load_json_file(file, [&](const simplemc::json_serializer & s){ s.load_at("m", fresh); }, checkpoint::io);
    fs::remove(file);
    const auto a {m.normalised()}, b {fresh.normalised()};
    const auto a3 {m.normalised3d()}, b3 {fresh.normalised3d()};
    const bool same {a.Z == b.Z && a.Z_error == b.Z_error && a.radial == b.radial && a3.weight == b3.weight && a3.error == b3.error};
    check(same && fresh.k_scale == m.k_scale && fresh.n_measured == m.n_measured && fresh.n_negative == m.n_negative,
          "checkpoint round trip: identical results, counters and k_s (%.2f)", fresh.k_scale);

    // an inactive (smallest-layout) estimator's empty checkpoint loads into a full one without error
    EM tiny {&c.d.cfg, p, 0., false, 2, 1.0, 2, 2, 1, 2};
    const fs::path f2 {fs::temp_directory_path() / "dmc_test_ckpt_electron_momentum_tiny.msgpack"};
    simplemc::save_json_file(f2, [&](simplemc::json_serializer & s){ s.save_at("m", tiny); }, checkpoint::io);
    EM full {&c.d.cfg, p, 0., false, 16, 0.8, 4, 8, 3, 4};
    bool loaded {true};
    try { simplemc::load_json_file(f2, [&](const simplemc::json_serializer & s){ s.load_at("m", full); }, checkpoint::io); }
    catch (const std::exception &) { loaded = false; }
    fs::remove(f2);
    check(loaded && full.acc.count() == 0 && full.k_scale == 0.8, "empty smallest-layout checkpoint: ignored by a full estimator");
}

int main(){
    test::Checks check {"electron_momentum: time-averaged n(k) per phonon sector"};
    test::setLK(test::AlAs);
    maps(check);
    sectors(check);
    bookkeeping(check);
    round_trip(check);
    return check.exit_code();
}
