#ifndef GREEN_TRACE_HPP
#define GREEN_TRACE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include "diagram/diagram_config.hpp"

// Binned estimator of the trace of the Green function, tr G(p, tau) = sum_n G_nn(p, tau) - or, selecting
// phonons present at the imaginary-time boundary, of the trace of the corresponding P function - with
// jackknife error bars.
//
// The chain samples diagrams with weight |W|, and sign(W) = sign(tr P) (P = diagram_head->right_component,
// everything else in W is positive). A diagram contributes W to tr G at its length tau_D, so each sample
// contributes sign(tr P) = +-1 to the bin containing tau_D: no band matrix enters. This is the trace column
// of green_func_measurement (whose components sum to P_nn / |tr P| = sign(tr P)) without the components.
//
// Everything else is as in green_func_measurement: e^{mu tau_D} is removed per sample as
// e^{-mu (tau_D - tau_b)} (tau_b the left edge of the bin) when accumulating and as e^{-mu tau_b}
// (measurement::binScale) after the jackknife; boundary_phonons selects the sector (0: no external lines, i.e.
// tr G(p, tau); n > 0: exactly n; < 0: all, the P function); and the normalisation is
//     tr G(bin) = Z_0 / bin_width * <x_bin> / <x_0>
// with x_0 the order-0 indicator, counted whatever the selection, and Z_0 its exact integral.
//
// split_sign = true additionally keeps the positive and negative parts separately: the samples with
// sign +1 and those with sign -1 go to two histograms, both stored as non-negative magnitudes and
// normalised in the same way, so that
//     tr G = positive - negative
// for every bin (and the trace is then computed as that difference of batch means). Their ratio measures
// the sign problem bin by bin. With split_sign = false only the signed histogram is accumulated.
//
// Statistics: one vector sample per measure(), accumulated sparsely in a simplemc batch accumulator -
// size n_bins + 1 (signed histogram, then the order-0 indicator), or 2*n_bins + 1 with split_sign
// (positive histogram, negative histogram, order-0 indicator). Delete-one-batch jackknife of the ratio
// above; at least 2 full batches are needed, i.e. at least ~ n_batches samples.
//
// Several ranks: collect the accumulator with simplemc_mpi_collect(comm, acc), and sum hits,
// n_selected, n_negative and n_order0, before calling normalised()/write().
struct green_trace_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const int boundary_phonons {0};
    const int n_bins {0};
    const bool split_sign {false};
    const double tau_max {0.};
    const double bin_width {0.};

    acc_type acc;                       // one vector sample per measure(); count() = number of samples
    std::vector<std::uint64_t> hits;    // selected samples per bin (diagnostic)
    std::uint64_t n_selected {0};       // samples that entered a bin
    std::uint64_t n_negative {0};       // ... of which with sign -1
    std::uint64_t n_order0 {0};         // samples with no phonon line at all

    struct result {
        std::vector<long double> trace;
        std::vector<long double> trace_error;
        std::vector<long double> positive;          // empty unless split_sign
        std::vector<long double> positive_error;
        std::vector<long double> negative;          // as a magnitude: trace = positive - negative
        std::vector<long double> negative_error;
    };

    green_trace_measurement(diagram_cfg * cfg, std::array<double, 3> p, int n_bins, int boundary_phonons = 0,
                            bool split_sign = false, std::size_t n_batches = 256);

    void measure();

    // The estimator before binScale, as a function of the batch means x: entries [bin] the trace, and
    // with split_sign entries [n_bins + bin] the positive and [2*n_bins + bin] the negative part.
    Eigen::VectorXd unscaled(const Eigen::VectorXd & x) const;

    // Raw delete-one-batch jackknife of unscaled() - use it for covariances (multiply entry i by the
    // binScale of its bin). Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeUnscaled() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return unscaled(x); }, acc);
    }

    // means (the estimator applied to the full-run means) and jackknife errors, binScale applied
    result normalised() const;

    // one line per bin: tau_centre hits, trace and error, and with split_sign the positive and negative
    // parts with their errors; preceded by '#' metadata lines
    void write(const std::string & path) const;

private:
    int histograms() const { return split_sign ? 2 : 1; }
};

#endif // !GREEN_TRACE_HPP
