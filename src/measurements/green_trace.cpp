#include "measurements/green_trace.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include "measurements/measurement_utils.hpp"

green_trace_measurement::green_trace_measurement(diagram_cfg * cfg, std::array<double, 3> p, int n_bins, int boundary_phonons,
                                                 bool split_sign, std::size_t n_batches)
    : cfg(cfg),
      p(p),
      boundary_phonons(boundary_phonons),
      n_bins(n_bins),
      split_sign(split_sign),
      tau_max(cfg->tau_max),
      bin_width(cfg->tau_max / n_bins),
      acc(static_cast<acc_type::size_type>((split_sign ? 2 : 1)*n_bins + 1), n_batches),
      hits(static_cast<std::size_t>(n_bins), 0)
{
    assert(cfg != nullptr);
    assert(n_bins > 0);
}

void green_trace_measurement::measure(){
    // one sparse sample: at most one histogram entry plus the order-0 indicator
    std::array<double, 2> vals {};
    std::array<acc_type::size_type, 2> idxs {};
    std::size_t k {0};

    const int n_internal {cfg->internal_ph_manager->current_length / 2};
    const int n_external {cfg->external_ph_manager->current_length / 2};

    if (boundary_phonons < 0 || n_external == boundary_phonons) {
        const double trace {cfg->diagram_head->right_component.trace()};
        if (std::isfinite(trace) && trace != 0.) {        // a zero trace has probability zero
            const double tau {cfg->current_tau_length};
            const int bin {std::clamp(static_cast<int>(tau / bin_width), 0, n_bins - 1)};
            // the bin-relative part of e^{-mu tau_D} (at most e^{|mu| bin_width})
            const double factor {std::exp(-cfg->chem_pot * (tau - bin * bin_width))};
            const bool negative {trace < 0.};
            if (split_sign) {
                vals[k] = factor;
                idxs[k] = (negative ? n_bins : 0) + bin;
            }
            else {
                vals[k] = negative ? -factor : factor;
                idxs[k] = bin;
            }
            ++k;
            ++hits[static_cast<std::size_t>(bin)];
            ++n_selected;
            if (negative) { ++n_negative; }
        }
    }

    // counted whatever the selection: the normalisation always refers to the order-0 diagram
    if (n_internal == 0 && n_external == 0) {
        vals[k] = 1.;
        idxs[k] = histograms()*n_bins;
        ++k;
        ++n_order0;
    }

    // also called with k = 0: every measure() is one sample, contributing zero to all entries
    acc.accumulate(std::span(vals.data(), k), std::span(idxs.data(), k));
}

Eigen::VectorXd green_trace_measurement::unscaled(const Eigen::VectorXd & x) const {
    const double norm {static_cast<double>(measurement::order0Integral(p, cfg->chem_pot, tau_max)) / bin_width / x(histograms()*n_bins)};
    if (!split_sign) {
        return norm * x.head(n_bins);
    }
    Eigen::VectorXd out(3*n_bins);
    for (int b {0}; b < n_bins; ++b) {
        out(n_bins + b) = norm * x(b);               // positive part
        out(2*n_bins + b) = norm * x(n_bins + b);    // negative part, as a magnitude
        out(b) = out(n_bins + b) - out(2*n_bins + b);
    }
    return out;
}

green_trace_measurement::result green_trace_measurement::normalised() const {
    const auto jk {jackknifeUnscaled()};
    const Eigen::VectorXd mean {jk.naive_mean()};    // the estimator applied to the full-run means
    const Eigen::VectorXd err {jk.stderror()};

    const auto nb {static_cast<std::size_t>(n_bins)};
    result r;
    r.trace.resize(nb);
    r.trace_error.resize(nb);
    if (split_sign) {
        r.positive.resize(nb);
        r.positive_error.resize(nb);
        r.negative.resize(nb);
        r.negative_error.resize(nb);
    }
    for (int b {0}; b < n_bins; ++b) {
        const long double s {measurement::binScale(b, bin_width, cfg->chem_pot)};
        const auto bb {static_cast<std::size_t>(b)};
        r.trace[bb] = s * mean(b);
        r.trace_error[bb] = s * err(b);
        if (split_sign) {
            r.positive[bb] = s * mean(n_bins + b);
            r.positive_error[bb] = s * err(n_bins + b);
            r.negative[bb] = s * mean(2*n_bins + b);
            r.negative_error[bb] = s * err(2*n_bins + b);
        }
    }
    return r;
}

void green_trace_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("green_trace_measurement::write: cannot open " + path); }

    std::fprintf(f, "# Green function trace estimator\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   boundary_phonons = %d%s\n", p[0], p[1], p[2], boundary_phonons,
                 boundary_phonons == 0 ? " (vacuum sector: tr G(p,tau))" : boundary_phonons < 0 ? " (all: P function)" : "");
    std::fprintf(f, "# tau_max = %.10f   n_bins = %d   bin_width = %.10f   mu = %.10f\n", tau_max, n_bins, bin_width, cfg->chem_pot);
    std::fprintf(f, "# n_samples = %llu   n_selected = %llu   n_negative = %llu   n_order0 = %llu   Z_0 = %.18Le   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_selected),
                 static_cast<unsigned long long>(n_negative), static_cast<unsigned long long>(n_order0), measurement::order0Integral(p, cfg->chem_pot, tau_max),
                 acc.batches().size(), static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, split_sign ? "# columns: tau_centre hits  trace err  positive err  negative err   (trace = positive - negative)\n"
                               : "# columns: tau_centre hits  trace err\n");
    for (int b {0}; b < n_bins; ++b) {
        const auto bb {static_cast<std::size_t>(b)};
        std::fprintf(f, "%.10Le %llu %.15Le %.6Le", measurement::binCentre(b, bin_width), static_cast<unsigned long long>(hits[bb]), r.trace[bb], r.trace_error[bb]);
        if (split_sign) {
            std::fprintf(f, " %.15Le %.6Le %.15Le %.6Le", r.positive[bb], r.positive_error[bb], r.negative[bb], r.negative_error[bb]);
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}
