#include "measurements/eff_mass_uniaxial.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "comp_method/weight_computation.hpp"
#include "measurements/measurement_utils.hpp"

namespace {
    // the six independent entries of a symmetric 3x3 matrix, in the accumulator's order
    constexpr std::array<std::array<int, 2>, 6> pairs {{{0,0}, {1,1}, {2,2}, {0,1}, {0,2}, {1,2}}};
}

bool eff_mass_uniaxial_measurement::applicable(){
    return weight::LKMatrix::C_LK == 0.;
}

eff_mass_uniaxial_measurement::eff_mass_uniaxial_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min,
                                                             bool tau_fixed, std::size_t n_batches)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      m_par(1. / (2. * weight::LKMatrix::A_LK)),
      m_perp(1. / (2. * weight::LKMatrix::B_LK)),
      tau_length(tau_fixed ? cfg->current_tau_length : -1.),
      acc(31, n_batches)
{
    assert(cfg != nullptr);
    assert(applicable());
}

void eff_mass_uniaxial_measurement::measure(){
    if (!tau_fixed && cfg->current_tau_length < tau_min) {
        return;
    }
    const Eigen::Matrix3d & P {cfg->diagram_head->right_component};
    const double trace {P.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    const double s {trace < 0. ? -1. : 1.};
    const double tau {cfg->current_tau_length};
    assert(!tau_fixed || tau == tau_length);       // tau_fixed: the length must not change during the run
    const Eigen::Vector3d Y {measurement::pathMomentum(*cfg)};

    // orbital basis: M_ii = s w_i (M is diagonal for C = 0, U being a signed permutation)
    const Eigen::Matrix3d & U {cfg->diagram_head->baseWF};
    const Eigen::Matrix3d M {U * P * U.transpose() / std::abs(trace)};

    Eigen::Matrix<double, 31, 1> sample;
    for (int i {0}; i < 3; ++i) {
        const double w {M(i, i)};
        sample(10*i) = w;
        sample.segment<3>(10*i + 1) = w * Y / tau;
        for (int c {0}; c < 6; ++c) {
            sample(10*i + 4 + c) = w * Y(pairs[c][0]) * Y(pairs[c][1]) / tau;
        }
    }
    sample(30) = s;

    acc.accumulate(sample);
    ++n_measured;
    if (s < 0.) { ++n_negative; }
}

Eigen::VectorXd eff_mass_uniaxial_measurement::ratios(const Eigen::VectorXd & x) const {
    const double sign {x(30)};
    // <Y><Y>/tau = tau_D <Y/tau><Y/tau> at fixed tau_D; with varying tau_D the subtraction is dropped (p = 0 only)
    const double subtract {tau_fixed ? tau_length : 0.};

    Eigen::VectorXd out(70);
    double w_sum {0.}, par {0.}, perp {0.};
    for (int i {0}; i < 3; ++i) {
        const double norm {x(10*i)};                                // <M_ii>: the sector's own normalisation
        const Eigen::Vector3d Y_tau {x.segment<3>(10*i + 1) / norm};  // <Y / tau>_i
        Eigen::Matrix3d C;
        for (int c {0}; c < 6; ++c) {
            const int a {pairs[c][0]}, b {pairs[c][1]};
            C(a, b) = x(10*i + 4 + c) / norm - subtract * Y_tau(a) * Y_tau(b);
            C(b, a) = C(a, b);
        }

        Eigen::Matrix3d inv_mass;
        for (int a {0}; a < 3; ++a) {
            for (int b {0}; b < 3; ++b) {
                inv_mass(a, b) = (a == b ? 1. / bareMass(i, a) : 0.) - C(a, b) / (bareMass(i, a) * bareMass(i, b));
                out(3 + 9*i + 3*a + b) = inv_mass(a, b);
                out(43 + 9*i + 3*a + b) = C(a, b);
            }
            out(30 + 3*i + a) = Y_tau(a) / bareMass(i, a);
        }

        const double w {norm / sign};                               // <w_i>
        out(i) = w;
        w_sum += w;
        par += w * inv_mass(i, i);
        perp += w * 0.5 * (inv_mass.trace() - inv_mass(i, i));
    }
    par /= w_sum;
    perp /= w_sum;
    out(39) = par;
    out(40) = perp;
    out(41) = 1. / (m_par * par);
    out(42) = 1. / (m_perp * perp);
    return out;
}

eff_mass_uniaxial_measurement::result eff_mass_uniaxial_measurement::normalised() const {
    const auto jk {jackknifeRatios()};
    const Eigen::VectorXd mean {jk.naive_mean()};    // the estimator applied to the full-run means
    const Eigen::VectorXd err {jk.stderror()};

    result r;
    for (int i {0}; i < 3; ++i) {
        r.w(i) = mean(i);
        r.w_error(i) = err(i);
        for (int a {0}; a < 3; ++a) {
            for (int b {0}; b < 3; ++b) {
                r.inv_mass[i](a, b) = mean(3 + 9*i + 3*a + b);
                r.inv_mass_error[i](a, b) = err(3 + 9*i + 3*a + b);
                r.C[i](a, b) = mean(43 + 9*i + 3*a + b);
                r.C_error[i](a, b) = err(43 + 9*i + 3*a + b);
            }
            r.velocity(i, a) = mean(30 + 3*i + a);
            r.velocity_error(i, a) = err(30 + 3*i + a);
        }
    }
    r.inv_mass_par = mean(39);
    r.inv_mass_par_error = err(39);
    r.inv_mass_perp = mean(40);
    r.inv_mass_perp_error = err(40);
    r.mass_ratio_par = mean(41);
    r.mass_ratio_par_error = err(41);
    r.mass_ratio_perp = mean(42);
    r.mass_ratio_perp_error = err(42);
    return r;
}

void eff_mass_uniaxial_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("eff_mass_uniaxial_measurement::write: cannot open " + path); }

    std::fprintf(f, "# polaron effective masses, uniaxial LK model (C = 0), orbital-resolved\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   ", p[0], p[1], p[2]);
    if (tau_fixed) { std::fprintf(f, "tau_fixed = %.10f\n", tau_length); }
    else { std::fprintf(f, "tau_min = %.10f (varying tau_D: covariances without the <Y><Y> term, valid at p = 0 only)\n", tau_min); }
    std::fprintf(f, "# A = %.10f   B = %.10f   m_par = 1/(2A) = %.10f   m_perp = 1/(2B) = %.10f\n",
                 weight::LKMatrix::A_LK, weight::LKMatrix::B_LK, m_par, m_perp);
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_negative = %llu   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_negative), acc.batches().size(),
                 static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# orbital fractions <w_x> <w_y> <w_z> (value err)\n#   % .15e %.6e   % .15e %.6e   % .15e %.6e\n",
                 r.w(0), r.w_error(0), r.w(1), r.w_error(1), r.w(2), r.w_error(2));

    const auto matrix {[f](const std::string & name, const Eigen::Matrix3d & v, const Eigen::Matrix3d & e){
        std::fprintf(f, "# %s (rows: value err value err value err)\n", name.c_str());
        for (int a {0}; a < 3; ++a) {
            std::fprintf(f, "#   % .15e %.6e   % .15e %.6e   % .15e %.6e\n",
                         v(a, 0), e(a, 0), v(a, 1), e(a, 1), v(a, 2), e(a, 2));
        }
    }};
    const char axis[3] {'x', 'y', 'z'};
    for (int i {0}; i < 3; ++i) {
        matrix(std::string("orbital ") + axis[i] + ": inverse-mass tensor 1/m* (a.u.)", r.inv_mass[i], r.inv_mass_error[i]);
    }
    matrix("group velocities (row i: orbital i; columns x y z)", r.velocity, r.velocity_error);
    for (int i {0}; i < 3; ++i) {
        matrix(std::string("orbital ") + axis[i] + ": C = Cov(Y_a, Y_b) / tau (for the two-length slope recipe)", r.C[i], r.C_error[i]);
    }

    std::fprintf(f, "# columns: 1/m*_par err 1/m*_perp err m*_par/m_par err m*_perp/m_perp err   (pooled over orbitals; p = 0)\n");
    std::fprintf(f, "%.15Le %.6Le %.15Le %.6Le %.15Le %.6Le %.15Le %.6Le\n",
                 r.inv_mass_par, r.inv_mass_par_error, r.inv_mass_perp, r.inv_mass_perp_error,
                 r.mass_ratio_par, r.mass_ratio_par_error, r.mass_ratio_perp, r.mass_ratio_perp_error);
    std::fclose(f);
}
