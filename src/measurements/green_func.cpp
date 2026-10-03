#include "measurements/green_func.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include "comp_method/weight_computation.hpp"

green_func_measurement::green_func_measurement(diagram_cfg * cfg, std::array<double, 3> p, int n_bins, int boundary_phonons,
                                               std::size_t n_batches)
    : cfg(cfg),
      p(p),
      boundary_phonons(boundary_phonons),
      n_bins(n_bins),
      tau_max(cfg->tau_max),
      bin_width(cfg->tau_max / n_bins),
      acc(static_cast<acc_type::size_type>(9*n_bins + 1), n_batches),
      hits(static_cast<std::size_t>(n_bins), 0)
{
    assert(cfg != nullptr);
    assert(n_bins > 0);
}

void green_func_measurement::measure(){
    // one sparse sample: at most 9 component entries of one bin plus the order-0 indicator
    std::array<double, 10> vals {};
    std::array<acc_type::size_type, 10> idxs {};
    std::size_t k {0};

    const int n_internal {cfg->internal_ph_manager->current_length / 2};
    const int n_external {cfg->external_ph_manager->current_length / 2};

    if (boundary_phonons < 0 || n_external == boundary_phonons) {
        const Eigen::Matrix3d & P {cfg->diagram_head->right_component};
        const double trace {P.trace()};
        if (std::isfinite(trace) && trace != 0.) {        // a zero trace has probability zero
            const double tau {cfg->current_tau_length};
            const int bin {std::clamp(static_cast<int>(tau / bin_width), 0, n_bins - 1)};
            // P_nm / |tr P|, times the bin-relative part of e^{-mu tau_D} (at most e^{|mu| bin_width})
            const double factor {std::exp(-cfg->chem_pot * (tau - bin * bin_width)) / std::abs(trace)};
            for (int n {0}; n < 3; ++n) {
                for (int m {0}; m < 3; ++m) {
                    vals[k] = P(n, m) * factor;
                    idxs[k] = 9*bin + 3*n + m;
                    ++k;
                }
            }
            ++hits[static_cast<std::size_t>(bin)];
            ++n_selected;
        }
    }

    // counted whatever the selection: the normalisation always refers to the order-0 diagram
    if (n_internal == 0 && n_external == 0) {
        vals[k] = 1.;
        idxs[k] = 9*n_bins;
        ++k;
        ++n_order0;
    }

    // also called with k = 0: every measure() is one sample, contributing zero to all entries
    acc.accumulate(std::span(vals.data(), k), std::span(idxs.data(), k));
}

long double green_func_measurement::order0Integral() const {
    // order-0 weight: tr e^{-H_LK(p) tau} e^{mu tau} = sum_n e^{-(E_n - mu) tau}, E_n = |p|^2 lambda_n
    // (the masses the diagram uses are 1/(2 lambda_n), and E = k^2/(2m)); the band order is irrelevant here
    const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(p)};
    const long double p_sq {static_cast<long double>(p[0])*p[0] + static_cast<long double>(p[1])*p[1] + static_cast<long double>(p[2])*p[2]};
    long double z0 {0.0L};
    for (int n {0}; n < 3; ++n) {
        const long double rate {p_sq * static_cast<long double>(eig(0, n)) - static_cast<long double>(cfg->chem_pot)};
        assert(rate > 0.0L);   // E_n >= 0 and mu < 0
        z0 += -std::expm1(-rate * static_cast<long double>(tau_max)) / rate;
    }
    return z0;
}

long double green_func_measurement::binCentre(int bin) const {
    return (static_cast<long double>(bin) + 0.5L) * static_cast<long double>(bin_width);
}

long double green_func_measurement::binScale(int bin) const {
    return std::exp(-static_cast<long double>(cfg->chem_pot) * static_cast<long double>(bin) * static_cast<long double>(bin_width));
}

Eigen::VectorXd green_func_measurement::unscaled(const Eigen::VectorXd & x) const {
    const double norm {static_cast<double>(order0Integral()) / bin_width / x(9*n_bins)};
    Eigen::VectorXd out(10*n_bins);
    for (int b {0}; b < n_bins; ++b) {
        for (int i {0}; i < 9; ++i) { out(9*b + i) = norm * x(9*b + i); }
        out(9*n_bins + b) = out(9*b) + out(9*b + 4) + out(9*b + 8);
    }
    return out;
}

green_func_measurement::result green_func_measurement::normalised() const {
    const auto jk {jackknifeUnscaled()};
    const Eigen::VectorXd mean {jk.naive_mean()};    // the estimator applied to the full-run means
    const Eigen::VectorXd err {jk.stderror()};

    result r;
    r.mean.resize(static_cast<std::size_t>(n_bins));
    r.error.resize(static_cast<std::size_t>(n_bins));
    r.trace.resize(static_cast<std::size_t>(n_bins));
    r.trace_error.resize(static_cast<std::size_t>(n_bins));
    for (int b {0}; b < n_bins; ++b) {
        const long double s {binScale(b)};
        const auto bb {static_cast<std::size_t>(b)};
        for (int i {0}; i < 9; ++i) {
            r.mean[bb][static_cast<std::size_t>(i)] = s * mean(9*b + i);
            r.error[bb][static_cast<std::size_t>(i)] = s * err(9*b + i);
        }
        r.trace[bb] = s * mean(9*n_bins + b);
        r.trace_error[bb] = s * err(9*n_bins + b);
    }
    return r;
}

void green_func_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("green_func_measurement::write: cannot open " + path); }

    std::fprintf(f, "# matrix Green function estimator\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   boundary_phonons = %d%s\n", p[0], p[1], p[2], boundary_phonons,
                 boundary_phonons == 0 ? " (vacuum sector: G_nm(p,tau))" : boundary_phonons < 0 ? " (all: P function)" : "");
    std::fprintf(f, "# tau_max = %.10f   n_bins = %d   bin_width = %.10f   mu = %.10f\n", tau_max, n_bins, bin_width, cfg->chem_pot);
    std::fprintf(f, "# n_samples = %llu   n_selected = %llu   n_order0 = %llu   Z_0 = %.18Le   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_selected),
                 static_cast<unsigned long long>(n_order0), order0Integral(), acc.batches().size(),
                 static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# columns: tau_centre hits  G_00 err  G_01 err  G_02 err  G_10 err  G_11 err  G_12 err  G_20 err  G_21 err  G_22 err  trace err\n");
    for (int b {0}; b < n_bins; ++b) {
        const auto bb {static_cast<std::size_t>(b)};
        std::fprintf(f, "%.10Le %llu", binCentre(b), static_cast<unsigned long long>(hits[bb]));
        for (std::size_t i {0}; i < 9; ++i) { std::fprintf(f, " %.15Le %.6Le", r.mean[bb][i], r.error[bb][i]); }
        std::fprintf(f, " %.15Le %.6Le\n", r.trace[bb], r.trace_error[bb]);
    }
    std::fclose(f);
}
