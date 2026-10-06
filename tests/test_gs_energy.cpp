// gs_energy_measurement, the tau-scaling energy estimator.
//
// 1. Per sample, on diagrams from a chain with all updates: X * tau must equal -n - d ln W / d lambda,
//    with W(lambda) the diagram's weight with every time scaled by lambda (electronic trace rebuilt from
//    k, masses and durations; phonon propagators e^{-omega lambda l}; couplings unchanged), evaluated
//    by a central finite difference. This checks every term and sign of the estimator independently of
//    how it is assembled from the cached products.
// 2. Order 0 (chg_tau only): every diagram is the bare propagator, so the chain average of X over
//    tau_D >= tau_min is exactly
//        int sum_n E_n e^{-(E_n - mu) tau} / int sum_n e^{-(E_n - mu) tau},   tau in [tau_min, tau_max],
//    which the jackknifed estimate must reproduce within its errors.
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
#include "measurements/gs_energy.hpp"

// ln |W(lambda)| up to lambda-independent factors (couplings), with all times scaled by lambda
static double log_weight_scaled(const diagram_cfg & cfg, double lambda){
    Eigen::Matrix3d prod {Eigen::Matrix3d::Identity()};
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        const std::array<double, 3> e {v->electronEnergy()};
        const double d {lambda * (v->tau_next - v->tau)};
        prod = prod * v->vertex_wf_component
             * Eigen::Matrix3d(Eigen::Vector3d(std::exp(-e[0]*d), std::exp(-e[1]*d), std::exp(-e[2]*d)).asDiagonal());
    }
    double phonon {0.};
    const double tau {cfg.current_tau_length};
    for (int i {0}; i < cfg.internal_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg.internal_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (v->conj_vertex->tau - v->tau);
    }
    for (int i {0}; i < cfg.external_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg.external_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (tau - v->tau + v->conj_vertex->tau);
    }
    return std::log(std::abs(prod.trace())) - lambda * phonon;
}

static void per_sample(test::Checks & check){
    test::Diagram d {2024ULL, {0.11, -0.07, 0.05}, 24, 8};
    diagram_cfg * cfg {&d.cfg};
    add_int_ph_update addi {cfg, &d.rng};  rm_int_ph_update rmi {cfg, &d.rng};
    add_ext_ph_update adde {cfg, &d.rng};  rm_ext_ph_update rme {cfg, &d.rng};
    swp_ph_update swp {cfg, &d.rng};       mv_tau_update mv {cfg, &d.rng};
    chg_tau_update ch {cfg, &d.rng};       chg_ph_momentum chw {cfg, &d.rng};
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick {0, 7};
    gs_energy_measurement E {cfg, {0.11, -0.07, 0.05}, 0.};

    const double h {1e-5};
    double worst {0.};
    long checked {0}, with_lines {0}, with_external {0};
    for (long step {0}; step < 200000; ++step) {
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
        if (step % 10 != 0) { continue; }

        const int n {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};
        const double dlogw {(log_weight_scaled(*cfg, 1. + h) - log_weight_scaled(*cfg, 1. - h)) / (2.*h)};
        const double expected {-static_cast<double>(n) - dlogw};
        const double got {E.estimate() * cfg->current_tau_length};
        worst = std::max(worst, std::abs(got - expected) / std::max(1., std::abs(expected)));
        ++checked;
        if (n > 0) { ++with_lines; }
        if (cfg->external_ph_manager->current_length > 0) { ++with_external; }
    }
    check(worst < 1e-6 && with_lines > 1000 && with_external > 100,
          "X * tau vs -n - d ln W/d lambda (finite difference) on %ld diagrams (%ld with lines, %ld with external lines): worst deviation %.1e",
          checked, with_lines, with_external, worst);
}

static void order0(test::Checks & check){
    const std::array<double, 3> p {0.155, -0.31, 0.095};
    const double tau_min {2.0};
    const int CHAINS {6};
    const long SAMPLES {2000000};

    double num {0.}, den {0.}, mu {0.}, tau_max {0.};
    double chi2 {0.}, worst_pull {0.};
    for (int c {0}; c < CHAINS; ++c) {
        test::Diagram d {4100ULL + static_cast<unsigned long long>(c), p, 4, 4};
        chg_tau_update ch {&d.cfg, &d.rng};
        std::uniform_real_distribution<double> u {0., 1.};
        gs_energy_measurement E {&d.cfg, p, tau_min};
        for (long s {0}; s < SAMPLES; ++s) {
            const double r {ch.attempt()};
            if (r > 0. && u(d.rng) < r) { ch.accept(); }
            E.measure();
        }
        mu = d.cfg.chem_pot; tau_max = d.cfg.tau_max;

        if (c == 0) {
            // exact average over [tau_min, tau_max]: closed form per band
            const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(p)};
            const double p_sq {p[0]*p[0] + p[1]*p[1] + p[2]*p[2]};
            for (int b {0}; b < 3; ++b) {
                const double e {p_sq * eig(0, b)}, a {e - mu};
                const double z {(std::exp(-a*tau_min) - std::exp(-a*tau_max)) / a};
                num += e * z; den += z;
            }
        }
        const gs_energy_measurement::result R {E.normalised()};
        const double pull {static_cast<double>((R.energy - num/den) / R.energy_error)};
        chi2 += pull*pull;
        worst_pull = std::max(worst_pull, std::abs(pull));
    }
    check(chi2 / CHAINS < 3. && worst_pull < 4.,
          "order 0, tau_D >= %.1f: estimate vs exact %.6f over %d chains, chi^2/dof = %.2f, worst pull %.1f (mu = %.2f, tau_max = %.1f)",
          tau_min, num/den, CHAINS, chi2 / CHAINS, worst_pull, mu, tau_max);
}

int main(){
    test::Checks check {"gs_energy_measurement (tau-scaling estimator)"};
    test::setLK(test::AlAs);
    per_sample(check);
    order0(check);
    return check.exit_code();
}
