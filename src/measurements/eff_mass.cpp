#include "measurements/eff_mass.hpp"
#include <mpi.h>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "comp_method/weight_computation.hpp"
#include "measurements/measurement_utils.hpp"
#include "utils/numerical.hpp"

namespace {
    // the six independent entries of a symmetric 3x3 matrix, in the accumulator's order
    constexpr std::array<std::array<int, 2>, 6> pairs {{{0,0}, {1,1}, {2,2}, {0,1}, {0,2}, {1,2}}};
}

bool eff_mass_measurement::applicable(){
    return numerical::isEqual(weight::LKMatrix::A_LK, weight::LKMatrix::B_LK)
        && numerical::isEqual(weight::LKMatrix::C_LK, 0.);
}

eff_mass_measurement::eff_mass_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed,
                                           std::size_t n_batches)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      m(1. / (2. * weight::LKMatrix::A_LK)),
      tau_length(tau_fixed ? cfg->current_tau_length : -1.),
      acc(10, n_batches)
{
    assert(cfg != nullptr);
    // applicable() is checked in measure(): an inactive instance may be built for any band structure (main
    // constructs every estimator; input::validate refuses an active one that does not apply)
}

Eigen::Vector3d eff_mass_measurement::pathMomentum() const {
    return measurement::pathMomentum(*cfg);
}

void eff_mass_measurement::measure(){
    assert(applicable());
    if (!tau_fixed && cfg->current_tau_length < tau_min) {
        return;
    }
    const double trace {cfg->diagram_head->right_component.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    const double s {trace < 0. ? -1. : 1.};
    const double tau {cfg->current_tau_length};
    assert(!tau_fixed || tau == tau_length);       // tau_fixed: the length must not change during the run
    const Eigen::Vector3d Y {pathMomentum()};

    Eigen::Matrix<double, 10, 1> sample;
    sample.segment<3>(0) = s * Y / tau;
    for (int c {0}; c < 6; ++c) {
        sample(3 + c) = s * Y(pairs[c][0]) * Y(pairs[c][1]) / tau;
    }
    sample(9) = s;

    acc.accumulate(sample);
    ++n_measured;
    if (s < 0.) { ++n_negative; }
}

Eigen::VectorXd eff_mass_measurement::ratios(const Eigen::VectorXd & x) const {
    const double sign {x(9)};
    const Eigen::Vector3d Y_tau {x.segment<3>(0) / sign};        // <Y / tau>
    // <Y><Y>/tau = tau_D <Y/tau><Y/tau> at fixed tau_D; with varying tau_D the subtraction is dropped (p = 0 only)
    const double subtract {tau_fixed ? tau_length : 0.};

    Eigen::Matrix3d C;
    for (int c {0}; c < 6; ++c) {
        const int a {pairs[c][0]}, b {pairs[c][1]};
        C(a, b) = x(3 + c) / sign - subtract * Y_tau(a) * Y_tau(b);
        C(b, a) = C(a, b);
    }
    const Eigen::Matrix3d inv_mass {Eigen::Matrix3d::Identity() - C / m};
    const double iso {inv_mass.trace() / 3.};

    Eigen::VectorXd out(9 + 2 + 3 + 9);
    for (int a {0}; a < 3; ++a) {
        for (int b {0}; b < 3; ++b) {
            out(3*a + b) = inv_mass(a, b);
            out(14 + 3*a + b) = C(a, b);
        }
    }
    out(9) = iso;
    out(10) = 1. / iso;
    out.segment<3>(11) = Y_tau / m;
    return out;
}

eff_mass_measurement::result eff_mass_measurement::normalised() const {
    const auto jk {jackknifeRatios()};
    const Eigen::VectorXd mean {jk.naive_mean()};    // the estimator applied to the full-run means
    const Eigen::VectorXd err {jk.stderror()};

    result r;
    for (int a {0}; a < 3; ++a) {
        for (int b {0}; b < 3; ++b) {
            r.inv_mass(a, b) = mean(3*a + b);
            r.inv_mass_error(a, b) = err(3*a + b);
            r.C(a, b) = mean(14 + 3*a + b);
            r.C_error(a, b) = err(14 + 3*a + b);
        }
        r.velocity(a) = mean(11 + a);
        r.velocity_error(a) = err(11 + a);
    }
    r.inv_mass_iso = mean(9);
    r.inv_mass_iso_error = err(9);
    r.mass_ratio = mean(10);
    r.mass_ratio_error = err(10);
    return r;
}

void eff_mass_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("eff_mass_measurement::write: cannot open " + path); }

    std::fprintf(f, "# polaron effective mass, isotropic Froehlich limit (A = B, C = 0)\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   ", p[0], p[1], p[2]);
    if (tau_fixed) { std::fprintf(f, "tau_fixed = %.10f\n", tau_length); }
    else { std::fprintf(f, "tau_min = %.10f (varying tau_D: covariance without the <Y><Y> term, valid at p = 0 only)\n", tau_min); }
    std::fprintf(f, "# A = %.10f   m = 1/(2A) = %.10f\n", weight::LKMatrix::A_LK, m);
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_negative = %llu   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_negative), acc.batches().size(),
                 static_cast<unsigned long long>(acc.batch_count()));

    const auto matrix {[f](const char * name, const Eigen::Matrix3d & v, const Eigen::Matrix3d & e){
        std::fprintf(f, "# %s (rows: value err value err value err)\n", name);
        for (int a {0}; a < 3; ++a) {
            std::fprintf(f, "#   % .15e %.6e   % .15e %.6e   % .15e %.6e\n",
                         v(a, 0), e(a, 0), v(a, 1), e(a, 1), v(a, 2), e(a, 2));
        }
    }};
    matrix("inverse-mass tensor m/m*", r.inv_mass, r.inv_mass_error);
    std::fprintf(f, "# group velocity v (value err, x y z)\n#   % .15e %.6e   % .15e %.6e   % .15e %.6e\n",
                 r.velocity(0), r.velocity_error(0), r.velocity(1), r.velocity_error(1), r.velocity(2), r.velocity_error(2));
    matrix("C = Cov(Y_a, Y_b) / tau (for the two-length slope recipe)", r.C, r.C_error);

    std::fprintf(f, "# columns: m/m* err m*/m err   (isotropic part, tr(m/m*)/3)\n");
    std::fprintf(f, "%.15Le %.6Le %.15Le %.6Le\n", r.inv_mass_iso, r.inv_mass_iso_error, r.mass_ratio, r.mass_ratio_error);
    std::fclose(f);
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, eff_mass_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    simplemc::mpi::all_reduce_in_place(m.n_measured, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
}

std::vector<std::string> eff_mass_measurement::summaryNames(){
    return {"mass_ratio", "inv_mass_iso"};
}

std::vector<double> eff_mass_measurement::summary() const {
    try {
        const result r {normalised()};
        return {static_cast<double>(r.mass_ratio), static_cast<double>(r.mass_ratio_error),
                static_cast<double>(r.inv_mass_iso), static_cast<double>(r.inv_mass_iso_error)};
    }
    catch (const std::exception &) {      // fewer than 2 full batches on this rank
        return std::vector<double>(4, std::numeric_limits<double>::quiet_NaN());
    }
}
