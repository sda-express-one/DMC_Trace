// eff_mass_uniaxial_measurement, the orbital-resolved effective masses of the uniaxial LK model (C = 0).
//
// 1. Orbital fractions, per sample on a chain with all updates: U P U^T is diagonal and its diagonal, over
//    |tr P|, equals w_i = e^{-(A-B) K_i} / sum_j e^{-(A-B) K_j} with K_i = sum_segments k_i^2 dtau computed
//    from the path - the exact split of the trace into the three single-orbital sectors.
// 2. Per-orbital factorisation: each sector weight W_i(d), rebuilt from scratch with every segment's momentum
//    shifted by d, gives d ln W_i / d d_a = -Y_a / m_ia and d^2 ln W_i / d d_a d d_b = -delta_ab tau / m_ia.
// 3. Bookkeeping: the estimates are the ratios of the test's own sums over the samples in full batches (the
//    same chain replayed, every third band matrix negated), with tau_D varying and at a fixed tau_D.
// 4. A = B: the pooled masses equal eff_mass_measurement's isotropic estimate on the same chain.
// 5. Physics at weak coupling, alpha ~ 0.3: the two-length pooled masses against the lowest-order uniaxial
//    results of Guster et al., PRB 104, 235123 (2021), Eqs. (68)-(75), whose isotropic limit f -> 6 is
//    checked first; for A = 2, B = 1 (mu* = m_perp / m_par = 2) and for the AlAs conduction band
//    (mu* = 0.271, the arcsin branch of S).
//
// Checks 1-3 use the AlAs conduction-band masses (A < B).
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
#include "measurements/eff_mass_uniaxial.hpp"
#include "measurements/measurement_utils.hpp"

// AlAs conduction band (X valley), Guster et al. Table S2: m_perp = 0.243 across the valley axis and
// m_z = 0.897 along it; here m_par = 1/(2A) = 0.897, m_perp = 1/(2B) = 0.243 (A < B, mu* = 0.271)
static constexpr double AlAs_m_par {0.897}, AlAs_m_perp {0.243};
static constexpr std::array<double, 3> AlAs_CB {1. / (2. * AlAs_m_par), 1. / (2. * AlAs_m_perp), 0.};

// A chain with every update; with fixed_tau the length-changing chg_tau is left out.
struct Chain {
    test::Diagram d;
    add_int_ph_update addi;  rm_int_ph_update rmi;
    add_ext_ph_update adde;  rm_ext_ph_update rme;
    swp_ph_update swp;       mv_tau_update mv;
    chg_tau_update ch;       chg_ph_momentum chw;
    std::uniform_real_distribution<double> u {0., 1.};
    std::uniform_int_distribution<int> pick;

    Chain(unsigned long long seed, std::array<double, 3> p, int max_int, int max_ext, double tau_max, double ph_energy,
          double diel, bool fixed_tau = false, double length = 1.)
        : d(seed, p, max_int, max_ext, tau_max, -1.0, ph_energy, diel),
          addi(&d.cfg, &d.rng), rmi(&d.cfg, &d.rng), adde(&d.cfg, &d.rng), rme(&d.cfg, &d.rng),
          swp(&d.cfg, &d.rng), mv(&d.cfg, &d.rng), ch(&d.cfg, &d.rng), chw(&d.cfg, &d.rng),
          pick(0, fixed_tau ? 6 : 7)
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

// U P U^T / |tr P| at the head: the orbital-basis diagram matrix of the current configuration
static Eigen::Matrix3d orbital_matrix(const diagram_cfg & cfg){
    const Eigen::Matrix3d & P {cfg.diagram_head->right_component};
    const Eigen::Matrix3d & U {cfg.diagram_head->baseWF};
    return U * P * U.transpose() / std::abs(P.trace());
}

// ln W_i for the three orbitals, rebuilt from scratch with every segment's momentum shifted by d
static Eigen::Vector3d log_sector_weights_shifted(const diagram_cfg & cfg, const Eigen::Vector3d & d){
    Eigen::Matrix3d prod {Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d U_prev {Eigen::Matrix3d::Identity()}, U_head {Eigen::Matrix3d::Identity()};
    bool first {true};
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        const std::array<double, 3> k {v->k[0] + d(0), v->k[1] + d(1), v->k[2] + d(2)};
        const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(k)};
        const Eigen::Matrix3d U {eig.block<3,3>(1,0)};
        const double k_sq {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};
        const double t {v->tau_next - v->tau};
        Eigen::Vector3d a;
        for (int n {0}; n < 3; ++n) { a(n) = std::exp(-k_sq * eig(0, n) * t); }   // E_n = k^2 lambda_n
        if (first) { U_head = U; } else { prod = prod * (U_prev.transpose() * U); }
        prod = prod * Eigen::Matrix3d(a.asDiagonal());
        U_prev = U;
        first = false;
    }
    const Eigen::Matrix3d W {U_head * prod * U_head.transpose()};
    return Eigen::Vector3d(std::log(W(0,0)), std::log(W(1,1)), std::log(W(2,2)));
}

static void orbital_fractions(test::Checks & check){
    const double A {weight::LKMatrix::A_LK}, B {weight::LKMatrix::B_LK};
    Chain c {11ULL, {0.11, -0.07, 0.05}, 24, 8, 10., 0.5, 2.};
    diagram_cfg * cfg {&c.d.cfg};
    double worst_off {0.}, worst_w {0.}, max_spread {0.};
    long checked {0};
    for (long step {0}; step < 200000; ++step) {
        c.step();
        if (step % 10 != 0) { continue; }
        const Eigen::Matrix3d M {orbital_matrix(*cfg)};
        Eigen::Vector3d K {Eigen::Vector3d::Zero()};
        for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
            for (int i {0}; i < 3; ++i) { K(i) += v->k[i] * v->k[i] * (v->tau_next - v->tau); }
        }
        const double K_min {K.minCoeff()};
        Eigen::Vector3d w;
        for (int i {0}; i < 3; ++i) { w(i) = std::exp(-(A - B) * (K(i) - K_min)); }
        w /= w.sum();
        worst_w = std::max(worst_w, (M.diagonal() - w).cwiseAbs().maxCoeff());
        worst_off = std::max({worst_off, std::abs(M(0,1)), std::abs(M(0,2)), std::abs(M(1,2)),
                                         std::abs(M(1,0)), std::abs(M(2,0)), std::abs(M(2,1))});
        max_spread = std::max(max_spread, w.maxCoeff() - w.minCoeff());
        ++checked;
    }
    check(worst_off < 1e-14 && worst_w < 1e-10,
          "U P U^T diagonal (worst off-diagonal %.1e) with w_i = e^{-(A-B)K_i}/sum (worst %.1e) on %ld diagrams; largest w spread %.3f",
          worst_off, worst_w, checked, max_spread);
}

static void per_orbital_factorisation(test::Checks & check){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    Chain c {2025ULL, p, 24, 8, 10., 0.5, 2.};
    diagram_cfg * cfg {&c.d.cfg};
    const eff_mass_uniaxial_measurement M {cfg, p, 0.};
    // ln W_i is exactly quadratic in the shift (C = 0), so central differences carry no truncation error
    const double h {1e-2};

    // the rebuild uses full (not band-normalised) actions, so a strongly suppressed orbital's weight can
    // underflow to 0 at some shift: such diagrams are skipped (counted), every other is checked in full
    const auto all_finite {[](std::initializer_list<Eigen::Vector3d> fs){
        for (const Eigen::Vector3d & f : fs) { if (!f.allFinite()) { return false; } }
        return true;
    }};
    double worst1 {0.}, worst2 {0.};
    long checked {0}, with_external {0}, skipped {0};
    for (long step {0}; step < 100000; ++step) {
        c.step();
        if (step % 20 != 0) { continue; }
        const Eigen::Vector3d Y {measurement::pathMomentum(*cfg)};
        const double tau {cfg->current_tau_length};
        const Eigen::Vector3d f0 {log_sector_weights_shifted(*cfg, Eigen::Vector3d::Zero())};
        std::array<Eigen::Vector3d, 3> fp, fm;
        std::array<Eigen::Vector3d, 3> mixed;       // (a, b) = (0,1), (0,2), (1,2)
        bool finite {f0.allFinite()};
        for (int a {0}; a < 3 && finite; ++a) {
            const Eigen::Vector3d ea {h * Eigen::Vector3d::Unit(a)};
            fp[a] = log_sector_weights_shifted(*cfg, ea);
            fm[a] = log_sector_weights_shifted(*cfg, -ea);
            finite = all_finite({fp[a], fm[a]});
        }
        for (int a {0}, n {0}; a < 3 && finite; ++a) {
            for (int b {a + 1}; b < 3 && finite; ++b, ++n) {
                const Eigen::Vector3d ea {h * Eigen::Vector3d::Unit(a)}, eb {h * Eigen::Vector3d::Unit(b)};
                const Eigen::Vector3d pp {log_sector_weights_shifted(*cfg, ea + eb)}, pm {log_sector_weights_shifted(*cfg, ea - eb)};
                const Eigen::Vector3d mp {log_sector_weights_shifted(*cfg, eb - ea)}, mm {log_sector_weights_shifted(*cfg, -ea - eb)};
                finite = all_finite({pp, pm, mp, mm});
                mixed[static_cast<std::size_t>(n)] = (pp - pm - mp + mm) / (4.*h*h);
            }
        }
        if (!finite) { ++skipped; continue; }

        for (int a {0}, n {0}; a < 3; ++a) {
            for (int i {0}; i < 3; ++i) {
                const double m {M.bareMass(i, a)};
                worst1 = std::max(worst1, std::abs((fp[a](i) - fm[a](i)) / (2.*h) + Y(a) / m) / std::max(1., std::abs(Y(a) / m)));
                worst2 = std::max(worst2, std::abs((fp[a](i) - 2.*f0(i) + fm[a](i)) / (h*h) + tau / m) / (tau / m));
            }
            for (int b {a + 1}; b < 3; ++b, ++n) {
                for (int i {0}; i < 3; ++i) {
                    worst2 = std::max(worst2, std::abs(mixed[static_cast<std::size_t>(n)](i)) / (tau / M.bareMass(i, a)));
                }
            }
        }
        ++checked;
        if (cfg->external_ph_manager->current_length > 0) { ++with_external; }
    }
    check(worst1 < 1e-8 && worst2 < 1e-6 && with_external > 0 && checked > 10 * skipped,
          "per orbital: d ln W_i/dd = -Y/m_i (worst %.1e), d^2 ln W_i/dd dd = -delta tau/m_i (worst %.1e) on %ld diagrams, %ld with external lines (%ld skipped: an orbital weight underflows)",
          worst1, worst2, checked, with_external, skipped);
}

// ratios() rebuilt from the test's own per-orbital sums
static Eigen::VectorXd expected_ratios(const eff_mass_uniaxial_measurement & M, double sum_s, const Eigen::Vector3d & sum_w,
                                       const std::array<Eigen::Vector3d, 3> & sum_wY, const std::array<Eigen::Matrix3d, 3> & sum_wYY,
                                       double subtract){
    Eigen::VectorXd out(70);
    double w_sum {0.}, par {0.}, perp {0.};
    for (int i {0}; i < 3; ++i) {
        const Eigen::Vector3d y {sum_wY[i] / sum_w(i)};
        const Eigen::Matrix3d C {sum_wYY[i] / sum_w(i) - subtract * y * y.transpose()};
        Eigen::Matrix3d inv;
        for (int a {0}; a < 3; ++a) {
            for (int b {0}; b < 3; ++b) {
                inv(a, b) = (a == b ? 1. / M.bareMass(i, a) : 0.) - C(a, b) / (M.bareMass(i, a) * M.bareMass(i, b));
                out(3 + 9*i + 3*a + b) = inv(a, b);
                out(43 + 9*i + 3*a + b) = C(a, b);
            }
            out(30 + 3*i + a) = y(a) / M.bareMass(i, a);
        }
        const double w {sum_w(i) / sum_s};
        out(i) = w;
        w_sum += w;
        par += w * inv(i, i);
        perp += w * 0.5 * (inv.trace() - inv(i, i));
    }
    out(39) = par / w_sum;
    out(40) = perp / w_sum;
    out(41) = 1. / (M.m_par * out(39));
    out(42) = 1. / (M.m_perp * out(40));
    return out;
}

static void bookkeeping(test::Checks & check, bool fixed){
    const std::array<double, 3> p {0.11, -0.07, 0.05};
    const long n_steps {600000};
    const double length {fixed ? 6. : 1.};

    std::uint64_t n_used {0}, n_negative {0};
    Eigen::VectorXd got;
    {
        Chain c {4242ULL, p, 24, 8, 10., 0.5, 2., fixed, length};
        diagram_cfg * cfg {&c.d.cfg};
        eff_mass_uniaxial_measurement M {cfg, p, 0., fixed};
        for (long step {0}; step < n_steps; ++step) {
            c.step();
            const bool flip {step % 3 == 0};
            if (flip) { cfg->diagram_head->right_component *= -1.; }
            M.measure();
            if (flip) { cfg->diagram_head->right_component *= -1.; }
        }
        got = M.jackknifeRatios().naive_mean();
        n_used = M.acc.batches().size() * M.acc.batch_count();
        n_negative = M.n_negative;
    }

    // the same chain replayed, summed independently over the samples in full batches
    double sum_s {0.};
    Eigen::Vector3d sum_w {Eigen::Vector3d::Zero()};
    std::array<Eigen::Vector3d, 3> sum_wY {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    std::array<Eigen::Matrix3d, 3> sum_wYY {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    Chain c {4242ULL, p, 24, 8, 10., 0.5, 2., fixed, length};
    diagram_cfg * cfg {&c.d.cfg};
    const eff_mass_uniaxial_measurement ref {cfg, p, 0., fixed};      // only for the masses and the fixed length
    for (long step {0}; step < static_cast<long>(n_used); ++step) {
        c.step();
        const double flip {step % 3 == 0 ? -1. : 1.};
        const Eigen::Matrix3d M {flip * orbital_matrix(*cfg)};
        const double tau {cfg->current_tau_length};
        Eigen::Vector3d Y {Eigen::Vector3d::Zero()};
        for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
            Y += (v->tau_next - v->tau) * Eigen::Vector3d(v->k[0], v->k[1], v->k[2]);
        }
        sum_s += flip * (cfg->diagram_head->right_component.trace() < 0. ? -1. : 1.);
        for (int i {0}; i < 3; ++i) {
            sum_w(i) += M(i, i);
            sum_wY[i] += M(i, i) * Y / tau;
            sum_wYY[i] += M(i, i) * Y * Y.transpose() / tau;
        }
    }
    const Eigen::VectorXd expected {expected_ratios(ref, sum_s, sum_w, sum_wY, sum_wYY, fixed ? length : 0.)};
    const double dev {((got - expected).array().abs() / expected.array().abs().max(1.)).maxCoeff()};
    check(dev < 1e-10 && n_negative > 0,
          "%s: all 70 estimates are the ratios of the test's own sums over the %llu samples in full batches (%llu negative): worst deviation %.1e",
          fixed ? "fixed tau_D = 6  " : "varying tau_D    ",
          static_cast<unsigned long long>(n_used), static_cast<unsigned long long>(n_negative), dev);
}

static void isotropic_limit(test::Checks & check){
    test::setLK({1., 1., 0.});
    const std::array<double, 3> p {0., 0., 0.};
    Chain c {606ULL, p, 24, 8, 10., 0.5, 2.};
    eff_mass_measurement Iso {&c.d.cfg, p, 0.};
    eff_mass_uniaxial_measurement Uni {&c.d.cfg, p, 0.};
    for (long step {0}; step < 400000; ++step) { c.step(); Iso.measure(); Uni.measure(); }
    const eff_mass_measurement::result ri {Iso.normalised()};
    const eff_mass_uniaxial_measurement::result ru {Uni.normalised()};
    const double iso {static_cast<double>(ri.inv_mass_iso) / Iso.m};
    double dev_tensor {0.};
    for (int i {0}; i < 3; ++i) { dev_tensor = std::max(dev_tensor, (ru.inv_mass[i] * Uni.m_par - ri.inv_mass).cwiseAbs().maxCoeff()); }
    const double dev {std::max({std::abs(static_cast<double>(ru.inv_mass_par) - iso), std::abs(static_cast<double>(ru.inv_mass_perp) - iso),
                                (ru.w - Eigen::Vector3d::Constant(1./3.)).cwiseAbs().maxCoeff()})};
    check(dev < 1e-10 && dev_tensor < 1e-10,
          "A = B: pooled 1/m*_par = 1/m*_perp = isotropic estimate (%.1e), every orbital tensor = isotropic tensor (%.1e), w_i = 1/3",
          dev, dev_tensor);
}

// Guster et al. Eq. (69): S(x) = arsinh(sqrt x)/sqrt x for x > 0, arcsin(sqrt(-x))/sqrt(-x) for x < 0
static double S_fn(double x){
    if (std::abs(x) < 1e-12) { return 1.; }
    return x > 0. ? std::asinh(std::sqrt(x)) / std::sqrt(x) : std::asin(std::sqrt(-x)) / std::sqrt(-x);
}
// Eqs. (74), (75), mu = m_perp / m_z
static double f_perp(double mu){ return 4. * (1. - mu) / (1. - std::sqrt(mu) / S_fn(mu - 1.)); }
static double f_z(double mu){ return 2. * (1. - mu) / mu / (1. / std::sqrt(mu) / S_fn(mu - 1.) - 1.); }

// Two-length pooled masses at p = 0 and weak coupling against Guster et al.'s lowest order, Eqs. (68)-(75),
// for bare masses m_par (an orbital's own axis, Guster's m_z) and m_perp; alpha set through eps.
static void weak_coupling_case(test::Checks & check, const char * label, double m_par, double m_perp, double eps,
                               unsigned long long seed){
    test::setLK({1. / (2. * m_par), 1. / (2. * m_perp), 0.});
    const double omega {1.};
    const double mu {m_perp / m_par};
    const double alpha {std::sqrt(m_perp) * S_fn(mu - 1.) / std::sqrt(2. * omega) / eps};      // Eqs. (46), (68)
    const double exp_par {1. / (1. - alpha / f_z(mu))}, exp_perp {1. / (1. - alpha / f_perp(mu))};

    const std::array<double, 3> p {0., 0., 0.};
    const std::array<double, 2> lengths {10., 20.};
    std::array<eff_mass_uniaxial_measurement::result, 2> r;
    for (int i {0}; i < 2; ++i) {
        Chain c {seed + static_cast<unsigned long long>(i), p, 200, 16, 30., omega, eps, true, lengths[i]};
        eff_mass_uniaxial_measurement M {&c.d.cfg, p, 0., true};
        for (long step {0}; step < 200000; ++step) { c.step(); }                 // thermalisation
        for (long step {0}; step < 4000000; ++step) { c.step(); M.measure(); }
        r[static_cast<std::size_t>(i)] = M.normalised();
    }

    // two-length slope on the pooled diagonal covariances: longitudinal C_i,ii and transverse C_i,aa (a != i),
    // weighted by <w_i> as in the pooled masses
    const auto pooled_C {[](const eff_mass_uniaxial_measurement::result & res, bool longitudinal){
        double v {0.}, e2 {0.}, ws {0.};
        for (int i {0}; i < 3; ++i) {
            const double w {res.w(i)};
            ws += w;
            if (longitudinal) { v += w * res.C[i](i, i); e2 += std::pow(w * res.C_error[i](i, i), 2); }
            else {
                for (int a {0}; a < 3; ++a) { if (a != i) { v += 0.5 * w * res.C[i](a, a); e2 += std::pow(0.5 * w * res.C_error[i](a, a), 2); } }
            }
        }
        return std::array<double, 2> {v / ws, std::sqrt(e2) / ws};     // conservative: errors added in quadrature
    }};
    const double dt {lengths[1] - lengths[0]};
    const auto slope {[&](bool longitudinal, double m){
        const std::array<double, 2> c1 {pooled_C(r[0], longitudinal)}, c2 {pooled_C(r[1], longitudinal)};
        const double inv {1. - (lengths[1]*c2[0] - lengths[0]*c1[0]) / (m * dt)};     // m/m*
        const double err {std::hypot(lengths[1]*c2[1], lengths[0]*c1[1]) / (m * dt)};
        return std::array<double, 2> {1. / inv, err / (inv * inv)};                    // m*/m and its error
    }};
    const std::array<double, 2> par {slope(true, m_par)}, perp {slope(false, m_perp)};
    check(std::abs(par[0] - exp_par) < 4.*par[1] + 5e-3 && std::abs(perp[0] - exp_perp) < 4.*perp[1] + 5e-3,
          "%s: alpha = %.3f, mu* = %.3f: two-length m*_par/m_par = %.4f +- %.4f vs %.4f, m*_perp/m_perp = %.4f +- %.4f vs %.4f (lowest order)",
          label, alpha, mu, par[0], par[1], exp_par, perp[0], perp[1], exp_perp);
}

static void weak_coupling(test::Checks & check){
    check(std::abs(f_perp(1. + 1e-6) - 6.) < 1e-4 && std::abs(f_z(1. + 1e-6) - 6.) < 1e-4,
          "Guster Eqs. (74)-(75): isotropic limit f_perp = %.6f, f_z = %.6f (both -> 6)", f_perp(1. + 1e-6), f_z(1. + 1e-6));
    weak_coupling_case(check, "A = 2, B = 1 (A > B)    ", 0.25, 0.5, 1.5, 9100ULL);
    weak_coupling_case(check, "AlAs conduction (A < B)", AlAs_m_par, AlAs_m_perp, 1.4, 9200ULL);
}

int main(){
    test::Checks check {"eff_mass_uniaxial_measurement (C = 0)"};
    test::setLK(AlAs_CB);
    check(eff_mass_uniaxial_measurement::applicable(), "applicable() for C = 0");
    test::setLK(test::AlAs);
    check(!eff_mass_uniaxial_measurement::applicable(), "not applicable() for C != 0");

    test::setLK(AlAs_CB);
    orbital_fractions(check);
    per_orbital_factorisation(check);
    bookkeeping(check, false);
    bookkeeping(check, true);
    isotropic_limit(check);
    weak_coupling(check);
    return check.exit_code();
}
