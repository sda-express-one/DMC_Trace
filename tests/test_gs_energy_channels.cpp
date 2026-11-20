// gs_energy_channels_measurement, the tau-scaling energy per symmetry channel of p, and the little-group channels
// (comp_method/little_group.hpp).
//
// 1. Channels: the order of G_p and the channel dimensions along [100], [110], [111] (and equivalent directions), on a
//    mirror plane, at a generic p and at p = 0; projectors idempotent, summing to 1 and commuting with G_p; the bare
//    lambda of each channel against the LK band coefficients.
// 2. Per sample, on AlAs diagrams from a chain with every update (external lines included), along [100] and [110]:
//    in the vacuum sector sum_c r_c = sign(tr P) and sum_c Y_c = sign(tr P) X of gs_energy; the segment products of
//    tau_scaling reproduce P with the action in place of the inserted energies; the counters split by sector.
// 3. Per channel, the scaling identity tau Y_c / r_c = -n - d ln |tr(Qt_c P(lambda))| / d lambda + sum omega l, by
//    central finite differences of the projected trace rebuilt from scratch.
// 4. Order 0 (bare propagator, chg_tau only): every channel along [100], [110], [111] has E_c = lambda_c p^2 exactly, per
//    sample, at every tau_D.
// 5. gs_energy::estimate() after its move onto tau_scaling is bitwise the original formula, per sample.
// 6. write(): one row per channel, read back.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
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
#include "comp_method/little_group.hpp"
#include "measurements/gs_energy.hpp"
#include "measurements/gs_energy_channels.hpp"
#include "measurements/tau_scaling.hpp"

static void channels(test::Checks & check){
    const double A {test::AlAs[0]}, B {test::AlAs[1]}, C {test::AlAs[2]};
    struct Case { const char * label; std::array<double, 3> p; int order; std::vector<int> dims; std::vector<std::vector<double>> bare; };
    const double s2 {std::sqrt(0.5)}, s3 {std::sqrt(1./3.)};
    const std::vector<Case> cases {
        {"p = 0", {0., 0., 0.}, 48, {3}, {{0., 0., 0.}}},
        {"[100]", {0.01, 0., 0.}, 8, {2, 1}, {{B, B}, {A}}},
        {"[0-10]", {0., -0.02, 0.}, 8, {2, 1}, {{B, B}, {A}}},
        {"[110]", {0.01*s2, 0.01*s2, 0.}, 4, {1, 1, 1}, {{(A + B - C)/2}, {B}, {(A + B + C)/2}}},
        {"[10-1]", {0.3, 0., -0.3}, 4, {1, 1, 1}, {{(A + B - C)/2}, {B}, {(A + B + C)/2}}},
        {"[111]", {0.01*s3, 0.01*s3, 0.01*s3}, 6, {2, 1}, {{(A + 2*B - C)/3, (A + 2*B - C)/3}, {(A + 2*B + 2*C)/3}}},
        {"[-111]", {-0.2, 0.2, 0.2}, 6, {2, 1}, {{(A + 2*B - C)/3, (A + 2*B - C)/3}, {(A + 2*B + 2*C)/3}}},
        {"mirror (0.3, 0.1, 0)", {0.3, 0.1, 0.}, 2, {}, {}},
        {"generic", {0.11, -0.07, 0.05}, 1, {3}, {}},
    };
    int ok_cases {0};
    double worst_proj {0.}, worst_bare {0.};
    for (const Case & c : cases) {
        const auto G {symmetry::littleGroup(c.p)};
        const auto ch {symmetry::channelsOf(c.p)};
        std::vector<int> dims;
        Eigen::Matrix3d sum {Eigen::Matrix3d::Zero()};
        for (const auto & x : ch) {
            dims.push_back(x.dim);
            sum += x.Q;
            worst_proj = std::max(worst_proj, (x.Q * x.Q - x.Q).cwiseAbs().maxCoeff());
            worst_proj = std::max(worst_proj, std::abs(x.Q.trace() - x.dim));
            for (const auto & R : G) { worst_proj = std::max(worst_proj, (x.Q * R - R * x.Q).cwiseAbs().maxCoeff()); }
        }
        worst_proj = std::max(worst_proj, (sum - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff());
        // the mirror plane: a 2-dim channel (the plane) and a 1-dim one (its normal, z), in either order
        const std::vector<int> want {c.dims.empty() ? std::vector<int>{} : c.dims};
        bool ok {static_cast<int>(G.size()) == c.order};
        if (c.dims.empty()) { ok = ok && dims.size() == 2 && dims[0] + dims[1] == 3; }
        else { ok = ok && dims == want; }
        for (std::size_t i {0}; i < c.bare.size() && ok; ++i) {
            for (std::size_t k {0}; k < c.bare[i].size(); ++k) {
                worst_bare = std::max(worst_bare, std::abs(ch[i].bare[k] - c.bare[i][k]));
            }
        }
        if (!ok) {
            std::printf("    %s: |G_p| = %zu, dims =", c.label, G.size());
            for (int d : dims) { std::printf(" %d", d); }
            std::printf("\n");
        }
        ok_cases += ok;
    }
    check(ok_cases == static_cast<int>(cases.size()) && worst_proj < 1e-12 && worst_bare < 1e-12,
          "little group and channels for %d momenta (p = 0, [100], [110], [111] and equivalents, mirror plane, generic): "
          "%d/%zu with the right |G_p| and dimensions; projectors (Q^2 = Q, sum = 1, [Q, R] = 0) worst %.1e; bare lambda worst %.1e",
          static_cast<int>(cases.size()), ok_cases, cases.size(), worst_proj, worst_bare);
}

// the original gs_energy formula (before tau_scaling), for the bitwise comparison
static double old_estimate(const diagram_cfg & cfg){
    const double tau {cfg.current_tau_length};
    const double trace {cfg.diagram_head->right_component.trace()};
    const int n_vertices {cfg.internal_ph_manager->current_length + cfg.external_ph_manager->current_length};
    double phonon {0.};
    for (int i {0}; i < cfg.internal_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg.internal_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (v->conj_vertex->tau - v->tau);
    }
    for (int i {0}; i < cfg.external_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg.external_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (tau - v->tau + v->conj_vertex->tau);
    }
    double electron {0.};
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        const std::array<double, 3> e {v->electronEnergy()};
        const Eigen::Matrix3d e_action {Eigen::Vector3d(e[0]*v->el_prop_action(0,0), e[1]*v->el_prop_action(1,1),
                                                        e[2]*v->el_prop_action(2,2)).asDiagonal()};
        electron += (v->tau_next - v->tau)
                  * (v->left_component * v->vertex_wf_component * e_action * v->next->vertex_wf_component * v->next->right_component).trace();
    }
    return (-static_cast<double>(n_vertices) + phonon + electron / trace) / tau;
}

// the electronic product with all durations scaled by lambda, rebuilt from scratch (head to tail), and sum omega l
static Eigen::Matrix3d product_scaled(const diagram_cfg & cfg, double lambda){
    Eigen::Matrix3d prod {Eigen::Matrix3d::Identity()};
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        const std::array<double, 3> e {v->electronEnergy()};
        const double d {lambda * (v->tau_next - v->tau)};
        prod = prod * v->vertex_wf_component
             * Eigen::Matrix3d(Eigen::Vector3d(std::exp(-e[0]*d), std::exp(-e[1]*d), std::exp(-e[2]*d)).asDiagonal());
    }
    return prod;
}

// a chain with every line update; f(step) is called on every 10th step
template <typename F>
static void run_chain(test::Diagram & d, long steps, F f){
    diagram_cfg * cfg {&d.cfg};
    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 7};
    for (long step {0}; step < steps; ++step) {
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
        if (step % 10 == 0) { f(step); }
    }
}

static void per_sample(test::Checks & check, const char * label, std::array<double, 3> p, unsigned long long seed){
    test::Diagram d {seed, p, 24, 8, 10., -1., 0.5, 2.0};
    diagram_cfg * cfg {&d.cfg};
    gs_energy_measurement E {cfg, p, 0.};
    gs_energy_channels_measurement M {cfg, p, 0.};
    const int nc {M.nChannels()};

    const double h {1e-5};
    double worst_rule {0.}, worst_prod {0.}, worst_fd {0.};
    long vacuum {0}, external {0}, with_lines {0}, fd_checked {0}, bitwise_diff {0};
    run_chain(d, 200000, [&](long){
        const double trace {cfg->diagram_head->right_component.trace()};
        if (!std::isfinite(trace) || trace == 0.) { return; }
        const double X {E.estimate()};
        if (X != old_estimate(*cfg)) { ++bitwise_diff; }
        M.measure();
        if (cfg->external_ph_manager->current_length > 0) { ++external; return; }
        ++vacuum;
        const int n {tau_scaling::nVertices(*cfg)};
        if (n > 0) { ++with_lines; }
        const double s {trace < 0. ? -1. : 1.};
        const Eigen::VectorXd y {M.estimate()};
        double sum_y {0.}, sum_r {0.};
        for (int c {0}; c < nc; ++c) { sum_y += y(2*c); sum_r += y(2*c + 1); }
        worst_rule = std::max({worst_rule, std::abs(sum_r - s), std::abs(sum_y - s * X) / std::max(1e-3, std::abs(X))});

        // the segment products with A in place of E A are P itself; the from-scratch product is P (same basis)
        const Eigen::Matrix3d & P {cfg->diagram_head->right_component};
        const double scale {P.cwiseAbs().maxCoeff()};
        for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
            const Eigen::Matrix3d S {v->left_component * v->vertex_wf_component * v->el_prop_action * v->next->vertex_wf_component
                                     * v->next->right_component};
            worst_prod = std::max(worst_prod, (S - P).cwiseAbs().maxCoeff() / scale);
        }

        // per channel: tau Y_c / r_c = -n + sum omega l - d ln|tr(Qt_c P(lambda))| / d lambda, with P(lambda) the full
        // (not band-normalised) product rebuilt with scaled durations, as in test_gs_energy's log_weight_scaled
        const Eigen::Matrix3d & U {cfg->diagram_head->baseWF};
        const double tau {cfg->current_tau_length};
        const Eigen::Matrix3d Pp {product_scaled(*cfg, 1. + h)}, Pm {product_scaled(*cfg, 1. - h)};
        const double phonon {tau_scaling::phononTerm(*cfg)};
        for (int c {0}; c < nc; ++c) {
            if (std::abs(y(2*c + 1)) < 1e-2) { continue; }
            const Eigen::Matrix3d Qt {U.transpose() * M.channels[static_cast<std::size_t>(c)].Q * U};
            const double dlog {(std::log(std::abs((Qt * Pp).trace())) - std::log(std::abs((Qt * Pm).trace()))) / (2.*h)};
            const double expected {-static_cast<double>(n) + phonon - dlog};
            const double got {tau * y(2*c) / y(2*c + 1)};
            worst_fd = std::max(worst_fd, std::abs(got - expected) / std::max(1., std::abs(expected)));
            ++fd_checked;
        }
    });
    check(worst_rule < 1e-10 && worst_prod < 1e-10 && vacuum > 5000 && external > 1000 && with_lines > 1000 && M.n_vacuum == static_cast<std::uint64_t>(vacuum)
          && M.n_measured == static_cast<std::uint64_t>(vacuum + external),
          "%s (%d channels): trace rule sum r_c = s, sum Y_c = s X on %ld vacuum diagrams (%ld with lines) worst %.1e; "
          "segment products = P worst %.1e; %ld diagrams with external lines counted, not projected (n_vacuum %llu, n_measured %llu)",
          label, nc, vacuum, with_lines, worst_rule, worst_prod, external,
          static_cast<unsigned long long>(M.n_vacuum), static_cast<unsigned long long>(M.n_measured));
    check(worst_fd < 1e-6 && fd_checked > 5000,
          "%s: per channel tau Y_c / r_c vs -n + sum omega l - d ln|tr(Qt_c P(lambda))|/d lambda (finite difference) on %ld "
          "(diagram, channel) pairs: worst deviation %.1e", label, fd_checked, worst_fd);
    check(bitwise_diff == 0, "%s: gs_energy::estimate() on tau_scaling is bitwise the original formula (%ld differences)",
          label, bitwise_diff);
}

static void order0(test::Checks & check){
    const double s2 {std::sqrt(0.5)}, s3 {std::sqrt(1./3.)};
    const std::vector<std::array<double, 3>> ps {{0.3, 0., 0.}, {0.3*s2, -0.3*s2, 0.}, {0.3*s3, 0.3*s3, -0.3*s3}};
    double worst {0.};
    long samples {0};
    int channels_seen {0};
    for (std::size_t k {0}; k < ps.size(); ++k) {
        const auto & p {ps[k]};
        test::Diagram d {900ULL + k, p, 4, 4};
        chg_tau_update ch {&d.cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        gs_energy_channels_measurement M {&d.cfg, p, 0.};
        const double p2 {p[0]*p[0] + p[1]*p[1] + p[2]*p[2]};
        channels_seen += M.nChannels();
        for (long s {0}; s < 20000; ++s) {
            const double r {ch.attempt()};
            if (r > 0. && u(d.rng) < r) { ch.accept(); }
            const Eigen::VectorXd y {M.estimate()};
            for (int c {0}; c < M.nChannels(); ++c) {
                const double want {M.channels[static_cast<std::size_t>(c)].bare.front() * p2};
                worst = std::max(worst, std::abs(y(2*c) / y(2*c + 1) - want) / want);
            }
            ++samples;
        }
    }
    check(worst < 1e-10 && channels_seen == 2 + 3 + 2,
          "order 0 along [100], [1-10], [11-1]: every channel's Y_c / r_c = lambda_c p^2 on %ld samples of every tau_D "
          "(%d channels): worst relative deviation %.1e", samples, channels_seen, worst);
}

static void file_round_trip(test::Checks & check){
    const std::array<double, 3> p {0.2*std::sqrt(0.5), 0.2*std::sqrt(0.5), 0.};
    test::Diagram d {77ULL, p, 24, 8, 10., -1., 0.5, 2.0};
    gs_energy_channels_measurement M {&d.cfg, p, 1., false, 16};
    run_chain(d, 200000, [&](long){ M.measure(); });
    const auto r {M.normalised()};
    const std::filesystem::path file {std::filesystem::temp_directory_path() / "dmc_test_gs_energy_channels.dat"};
    M.write(file.string());
    std::ifstream in {file};
    std::string line;
    int rows {0}, ok_rows {0};
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        std::istringstream ss {line};
        int c {0}, dim {0};
        long double e {0.L}, de {0.L}, sh {0.L}, dsh {0.L};
        ss >> c >> dim >> e >> de >> sh >> dsh;
        const auto i {static_cast<std::size_t>(c)};
        if (dim == M.channels[i].dim && std::abs(e - r.energy[i]) <= 1e-14L * std::abs(r.energy[i])
            && std::abs(sh - r.share[i]) <= 1e-9L * std::abs(r.share[i])) { ++ok_rows; }
        ++rows;
    }
    std::filesystem::remove(file);
    long double share_sum {0.L};
    for (const auto x : r.share) { share_sum += x; }
    check(rows == 3 && ok_rows == 3 && M.summary().size() == 6 && share_sum > 0.L && share_sum <= r.vacuum + 1e-12L,
          "[110], variable tau_D >= 1: write() gives %d rows (3 channels), %d matching normalised(); shares sum to %.4Lf "
          "<= vacuum fraction %.4Lf; summary() has %zu entries", rows, ok_rows, share_sum, r.vacuum, M.summary().size());
}

int main(){
    test::Checks check {"gs_energy_channels_measurement (tau-scaling energy per symmetry channel of p)"};
    test::setLK(test::AlAs);
    channels(check);
    per_sample(check, "AlAs [100]", {0.11, 0., 0.}, 2024ULL);
    per_sample(check, "AlAs [110]", {0.08, 0.08, 0.}, 2025ULL);
    per_sample(check, "AlAs generic", {0.11, -0.07, 0.05}, 2026ULL);
    order0(check);
    file_round_trip(check);
    return check.exit_code();
}
