// eff_mass_measurement, the effective-mass estimator of the isotropic Froehlich limit (A = B, C = 0).
//
// 1. Per sample, on diagrams from a chain with all updates at a generic p: with every segment's momentum
//    shifted by d, the diagram's weight is rebuilt from scratch (re-diagonalised bands, overlaps, actions;
//    couplings and phonon propagators do not depend on d) and finite differences must give
//        d ln W / d d_a = -Y_a / m,      d^2 ln W / d d_a d d_b = -delta_ab tau / m,
//    with Y = pathMomentum(). This checks the factorisation the estimator rests on, overlaps included.
// 2. Bookkeeping: the estimates must be the ratios of the test's own sums over the samples in full batches
//    (the same deterministic chain replayed), with every third band matrix negated to exercise the sign;
//    once with tau_D varying (no <Y><Y> subtraction) and once at a fixed tau_D (subtraction active).
// 3. p = 0: <Y> = 0, i.e. v = 0, and an isotropic tensor (off-diagonals 0, diagonals equal) within errors.
// 4. Physics at weak coupling: alpha = 0.5, two fixed lengths tau_D, the two-length slope recipe of the
//    header against m*/m = 1 + alpha/6 + 0.0236 alpha^2 (second-order perturbation theory).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
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
#include "measurements/eff_mass.hpp"

// A chain with every update; with fixed_tau the length-changing chg_tau is left out.
struct Chain {
    test::Diagram d;
    add_int_ph_update addi;  rm_int_ph_update rmi;
    add_ext_ph_update adde;  rm_ext_ph_update rme;
    swp_ph_update swp;       mv_tau_update mv;
    chg_tau_update ch;       chg_ph_momentum chw;
    const bool fixed_tau;
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick;

    Chain(unsigned long long seed, std::array<double, 3> p, int max_int, int max_ext, double tau_max, double ph_energy,
          double diel, bool fixed_tau = false, double length = 1.)
        : d(seed, p, max_int, max_ext, tau_max, -1.0, ph_energy, diel),
          addi(&d.cfg, &d.rng), rmi(&d.cfg, &d.rng), adde(&d.cfg, &d.rng), rme(&d.cfg, &d.rng),
          swp(&d.cfg, &d.rng), mv(&d.cfg, &d.rng), ch(&d.cfg, &d.rng), chw(&d.cfg, &d.rng),
          fixed_tau(fixed_tau), pick(0, fixed_tau ? 6 : 7)
    {
        if (length != d.cfg.current_tau_length) {
            // the order-0 diagram at the requested length, cached state rebuilt by the sanitizer
            d.cfg.current_tau_length = length;
            d.cfg.diagram_tail->tau = length;
            d.cfg.diagram_head->tau_next = length;
            numerical::sanitizeDiagram(&d.cfg);
        }
    }

    void step(){
        // 0..6 keep tau_D; 7 is chg_tau
        const int w {pick(d.rng)};
        double r {-1.};
        switch (w) { case 0: r = addi.attempt(); break; case 1: r = rmi.attempt(); break; case 2: r = adde.attempt(); break;
                     case 3: r = rme.attempt(); break; case 4: r = swp.attempt(); break; case 5: r = mv.attempt(); break;
                     case 6: r = chw.attempt(); break; default: r = ch.attempt(); }
        if (r > 0. && u(d.rng) < r) {
            switch (w) { case 0: addi.accept(); break; case 1: rmi.accept(); break; case 2: adde.accept(); break;
                         case 3: rme.accept(); break; case 4: swp.accept(); break; case 5: mv.accept(); break;
                         case 6: chw.accept(); break; default: ch.accept(); }
        }
    }
};

// ln |electronic trace| with every segment's momentum shifted by d, rebuilt from scratch
static double log_weight_shifted(const diagram_cfg & cfg, const Eigen::Vector3d & d){
    Eigen::Matrix3d prod {Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d U_prev {Eigen::Matrix3d::Identity()};
    bool first {true};
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        const std::array<double, 3> k {v->k[0] + d(0), v->k[1] + d(1), v->k[2] + d(2)};
        const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(k)};
        const Eigen::Matrix3d U {eig.block<3,3>(1,0)};
        const double k_sq {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};
        const double t {v->tau_next - v->tau};
        Eigen::Vector3d a;
        for (int n {0}; n < 3; ++n) { a(n) = std::exp(-k_sq * eig(0, n) * t); }   // E_n = k^2 lambda_n
        if (!first) { prod = prod * (U_prev.transpose() * U); }
        prod = prod * Eigen::Matrix3d(a.asDiagonal());
        U_prev = U;
        first = false;
    }
    return std::log(std::abs(prod.trace()));
}

static void per_sample(test::Checks & check){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    Chain c {2025ULL, p, 24, 8, 10., 0.5, 2.};
    diagram_cfg * cfg {&c.d.cfg};
    const eff_mass_measurement M {cfg, p, 0.};
    // ln W is exactly quadratic in the shift here (A = B, C = 0), so central differences carry no truncation
    // error and a large step only reduces rounding (~ 1e-16 |ln W| / h^2 in the second derivative)
    const double h {1e-2};

    double worst1 {0.}, worst2 {0.};
    long checked {0}, with_external {0};
    for (long step {0}; step < 100000; ++step) {
        c.step();
        if (step % 20 != 0) { continue; }
        const Eigen::Vector3d Y {M.pathMomentum()};
        const double tau {cfg->current_tau_length};
        const double f0 {log_weight_shifted(*cfg, Eigen::Vector3d::Zero())};
        for (int a {0}; a < 3; ++a) {
            const Eigen::Vector3d ea {h * Eigen::Vector3d::Unit(a)};
            const double fp {log_weight_shifted(*cfg, ea)}, fm {log_weight_shifted(*cfg, -ea)};
            worst1 = std::max(worst1, std::abs((fp - fm) / (2.*h) + Y(a) / M.m) / std::max(1., std::abs(Y(a) / M.m)));
            worst2 = std::max(worst2, std::abs((fp - 2.*f0 + fm) / (h*h) + tau / M.m) / (tau / M.m));
            for (int b {a + 1}; b < 3; ++b) {
                const Eigen::Vector3d eb {h * Eigen::Vector3d::Unit(b)};
                const double mixed {(log_weight_shifted(*cfg, ea + eb) - log_weight_shifted(*cfg, ea - eb)
                                   - log_weight_shifted(*cfg, eb - ea) + log_weight_shifted(*cfg, -ea - eb)) / (4.*h*h)};
                worst2 = std::max(worst2, std::abs(mixed) / (tau / M.m));
            }
        }
        ++checked;
        if (cfg->external_ph_manager->current_length > 0) { ++with_external; }
    }
    check(worst1 < 1e-6 && worst2 < 1e-4 && with_external > 0,
          "d ln W/dd = -Y/m (worst %.1e) and d^2 ln W/dd dd = -delta tau/m (worst %.1e) on %ld diagrams, %ld with external lines",
          worst1, worst2, checked, with_external);
}

// fixed: tau_D held at 6 (chg_tau left out) and tau_fixed, so the <Y><Y> subtraction is active; otherwise
// tau_D varies and C carries no subtraction
static void bookkeeping(test::Checks & check, bool fixed){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    const long n_steps {600000};
    const double length {fixed ? 6. : 1.};

    std::uint64_t n_used {0}, n_measured {0}, n_negative {0};
    eff_mass_measurement::result r;
    double m {0.};
    {
        Chain c {4242ULL, p, 24, 8, 10., 0.5, 2., fixed, length};
        diagram_cfg * cfg {&c.d.cfg};
        eff_mass_measurement M {cfg, p, 0., fixed};
        for (long step {0}; step < n_steps; ++step) {
            c.step();
            const bool flip {step % 3 == 0};
            if (flip) { cfg->diagram_head->right_component *= -1.; }
            M.measure();
            if (flip) { cfg->diagram_head->right_component *= -1.; }
        }
        r = M.normalised();
        n_used = M.acc.batches().size() * M.acc.batch_count();
        n_measured = M.n_measured;
        n_negative = M.n_negative;
        m = M.m;
    }

    // the same chain replayed, summed independently over the samples in full batches
    double sum_s {0.};
    Eigen::Vector3d sum_tau {Eigen::Vector3d::Zero()};
    Eigen::Matrix3d sum_YY {Eigen::Matrix3d::Zero()};
    {
        Chain c {4242ULL, p, 24, 8, 10., 0.5, 2., fixed, length};
        diagram_cfg * cfg {&c.d.cfg};
        for (long step {0}; step < static_cast<long>(n_used); ++step) {
            c.step();
            const double s {(step % 3 == 0 ? -1. : 1.) * (cfg->diagram_head->right_component.trace() < 0. ? -1. : 1.)};
            const double tau {cfg->current_tau_length};
            Eigen::Vector3d Y {Eigen::Vector3d::Zero()};
            for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
                Y += (v->tau_next - v->tau) * Eigen::Vector3d(v->k[0], v->k[1], v->k[2]);
            }
            sum_s += s;
            sum_tau += s * Y / tau;
            sum_YY += s * Y * Y.transpose() / tau;
        }
    }
    // Cov(Y)/tau: at fixed tau_D <Y><Y>/tau = tau_D <Y/tau><Y/tau>; with varying tau_D no subtraction
    const Eigen::Vector3d v_tau {sum_tau / sum_s};
    const Eigen::Matrix3d C {sum_YY / sum_s - (fixed ? length : 0.) * v_tau * v_tau.transpose()};
    const Eigen::Matrix3d inv_mass {Eigen::Matrix3d::Identity() - C / m};
    const double iso {inv_mass.trace() / 3.};

    const double dev {std::max({(r.inv_mass - inv_mass).cwiseAbs().maxCoeff(), (r.C - C).cwiseAbs().maxCoeff(),
                                (r.velocity - sum_tau / sum_s / m).cwiseAbs().maxCoeff(),
                                std::abs(static_cast<double>(r.inv_mass_iso) - iso),
                                std::abs(static_cast<double>(r.mass_ratio) - 1. / iso)})};
    check(dev < 1e-10 && n_measured == static_cast<std::uint64_t>(n_steps) && n_negative > 0,
          "%s: estimates are the ratios of the test's own sums over the %llu samples in full batches (%llu negative): worst deviation %.1e",
          fixed ? "fixed tau_D = 6  " : "varying tau_D    ",
          static_cast<unsigned long long>(n_used), static_cast<unsigned long long>(n_negative), dev);
}

static void isotropy(test::Checks & check){
    const std::array<double, 3> p {0., 0., 0.};
    Chain c {777ULL, p, 24, 8, 10., 0.5, 2.};
    eff_mass_measurement M {&c.d.cfg, p, 0.};
    for (long step {0}; step < 2000000; ++step) { c.step(); M.measure(); }
    const eff_mass_measurement::result r {M.normalised()};

    double pull_v {0.}, pull_off {0.}, pull_diag {0.};
    for (int a {0}; a < 3; ++a) {
        pull_v = std::max(pull_v, std::abs(r.velocity(a)) / r.velocity_error(a));
        for (int b {a + 1}; b < 3; ++b) {
            pull_off = std::max(pull_off, std::abs(r.inv_mass(a, b)) / r.inv_mass_error(a, b));
            pull_diag = std::max(pull_diag, std::abs(r.inv_mass(a, a) - r.inv_mass(b, b))
                                            / std::hypot(r.inv_mass_error(a, a), r.inv_mass_error(b, b)));
        }
    }
    check(pull_v < 5. && pull_off < 5. && pull_diag < 5.,
          "p = 0: v = 0 (worst %.2f sigma), off-diagonals 0 (%.2f sigma), equal diagonals (%.2f sigma); m/m* = %.4f +- %.4f (tau_D mixed, not converged)",
          pull_v, pull_off, pull_diag, static_cast<double>(r.inv_mass_iso), static_cast<double>(r.inv_mass_iso_error));
}

static void weak_coupling(test::Checks & check){
    // A = B = 1 (m = 1/2), omega = 1, eps = 1: alpha = (1/eps) sqrt(m/(2 omega)) = 0.5
    const double alpha {0.5};
    const std::array<double, 3> p {0., 0., 0.};
    const std::array<double, 2> lengths {10., 20.};
    std::array<eff_mass_measurement::result, 2> r;
    double m {0.};
    int max_order {0};
    for (int i {0}; i < 2; ++i) {
        Chain c {9000ULL + static_cast<unsigned long long>(i), p, 200, 16, 30., 1., 1., true, lengths[i]};
        eff_mass_measurement M {&c.d.cfg, p, 0., true};
        for (long step {0}; step < 200000; ++step) { c.step(); }                 // thermalisation
        for (long step {0}; step < 4000000; ++step) {
            c.step();
            M.measure();
            max_order = std::max(max_order, c.d.cfg.internal_ph_manager->current_length);
        }
        r[static_cast<std::size_t>(i)] = M.normalised();
        m = M.m;
    }

    // two-length slope recipe on the isotropic part: (m/m*) = 1 - (t2 c2 - t1 c1) / (m (t2 - t1)), c = tr C / 3
    const double c1 {r[0].C.trace() / 3.}, c2 {r[1].C.trace() / 3.};
    const double e1 {std::sqrt(r[0].C_error.diagonal().squaredNorm()) / 3.}, e2 {std::sqrt(r[1].C_error.diagonal().squaredNorm()) / 3.};
    const double dt {lengths[1] - lengths[0]};
    const double inv_mass {1. - (lengths[1]*c2 - lengths[0]*c1) / (m * dt)};
    const double inv_mass_err {std::hypot(lengths[1]*e2, lengths[0]*e1) / (m * dt)};   // conservative: diagonal errors added in quadrature
    const double expected {1. / (1. + alpha/6. + 0.0236*alpha*alpha)};
    check(std::abs(inv_mass - expected) < 4.*inv_mass_err + 5e-3 && max_order < 200,
          "alpha = %.1f: two-length m/m* = %.4f +- %.4f vs perturbation theory %.4f (single lengths: %.4f, %.4f); max internal vertices %d of 200",
          alpha, inv_mass, inv_mass_err, expected, static_cast<double>(r[0].inv_mass_iso), static_cast<double>(r[1].inv_mass_iso), max_order);
}

int main(){
    test::Checks check {"eff_mass_measurement (A = B, C = 0)"};
    test::setLK({1., 1., 0.});
    check(eff_mass_measurement::applicable(), "applicable() for A = B, C = 0");
    test::setLK(test::AlAs);
    check(!eff_mass_measurement::applicable(), "not applicable() for the AlAs parameters");
    test::setLK({1., 1., 0.});

    per_sample(check);
    bookkeeping(check, false);
    bookkeeping(check, true);
    isotropy(check);
    weak_coupling(check);
    return check.exit_code();
}
