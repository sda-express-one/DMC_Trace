#ifndef GREEN_FUNC_HPP
#define GREEN_FUNC_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include "diagram/diagram_config.hpp"

// Binned estimator of the band-matrix Green function G_nm(p, tau) - or, selecting phonons present at
// the imaginary-time boundary, of the corresponding P function - with jackknife error bars.
//
// The chain samples diagrams with weight |W|, W = tr(P) * (couplings^2, phonon propagators, e^{mu tau_D})
// where P = diagram_head->right_component is the diagram's band matrix. Everything multiplying tr(P) is
// positive, so sign(W) = sign(tr P), and a diagram contributes P_nm * (rest) = sign(W)|W| P_nm / tr(P)
// to component nm: each sample contributes P_nm / |tr P|. With today's positive-only sampling this is
// P_nm / tr P, and it is already the signed estimator once |W| is sampled.
//
// The factor e^{mu tau_D} is a sampling device only and is removed exactly, per sample, in two parts:
// e^{-mu (tau_D - tau_b)} with tau_b the left edge of the sample's bin is applied when accumulating
// (bounded by e^{|mu| bin_width}, so nothing large ever enters the statistics), and the per-bin
// constant e^{-mu tau_b} (binScale) is applied to the results after the jackknife, in long double.
//
// Boundary phonons: each external line is a phonon present at tau = 0 = tau_D, and with m of them the
// end segments carry p - sum(w), not p. boundary_phonons selects which diagrams are accumulated:
//   0      only diagrams without external lines: the vacuum sector, i.e. G_nm(p, tau)
//   1, 2.. only diagrams with exactly that many external lines
//   < 0    every diagram: the full trace over boundary states (P function)
// Indices n, m are bands in the order diagonalizeLKHamiltonian returns them at the end segments'
// momentum. For selection 0 that momentum is always p, so components are well defined (at a
// non-degenerate p). For any other selection it varies from diagram to diagram, so only the trace of
// the result is labelling-independent; the components are kept for inspection.
//
// Normalisation: the order-0 diagram (no phonon line at all) has the known weight
// sum_n e^{-(E_n(p) - mu) tau}, whose integral Z_0 over (0, tau_max] is exact. Order-0 samples are
// counted whatever the selection, and every selected sample comes from the same chain, so
//     G_nm(bin) = Z_0 / bin_width * <x_nm,bin> / <x_0>
// where x_nm,bin is a sample's contribution to that bin and component and x_0 its order-0 indicator.
//
// Statistics: every sample is one vector of size 9*n_bins + 1 - index 9*bin + 3*n + m for the
// components, the last index for the order-0 indicator - accumulated sparsely (at most 10 non-zero
// entries) in a simplemc batch accumulator. Batch means of batches longer than the autocorrelation
// time are close to independent, and the delete-one-batch jackknife of the ratio above then gives
// error bars that include the autocorrelation, the fluctuation of the order-0 count and the
// correlation between components - which is why the trace is estimated as its own function of the
// batch means rather than from the component errors (the components of a bin are correlated at
// ~ +1). At least 2 full batches are needed, i.e. at least ~ n_batches samples.
//
// Several ranks: collect the accumulator with simplemc_mpi_collect(comm, acc), and sum hits,
// n_selected and n_order0, before calling normalised()/write().
struct green_func_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const int boundary_phonons {0};
    const int n_bins {0};
    const double tau_max {0.};
    const double bin_width {0.};

    acc_type acc;                       // one vector sample per measure(); count() = number of samples
    std::vector<std::uint64_t> hits;    // selected samples per bin (diagnostic)
    std::uint64_t n_selected {0};       // samples that entered a bin
    std::uint64_t n_order0 {0};         // samples with no phonon line at all

    struct result {
        std::vector<std::array<long double, 9>> mean;    // per bin, row-major [3*n + m]
        std::vector<std::array<long double, 9>> error;   // jackknife standard error
        std::vector<long double> trace;
        std::vector<long double> trace_error;
    };

    green_func_measurement(diagram_cfg * cfg, std::array<double, 3> p, int n_bins, int boundary_phonons = 0,
                           std::size_t n_batches = 256);

    void measure();

    // Z_0 = int_0^tau_max tr G0(p, tau) e^{mu tau} dtau = sum_n (1 - e^{-(E_n - mu) tau_max}) / (E_n - mu)
    long double order0Integral() const;

    long double binCentre(int bin) const;
    long double binScale(int bin) const;   // e^{-mu tau_b}, tau_b = left edge of the bin

    // The estimator before binScale, as a function of the batch means x: entries [9*bin + 3*n + m] are
    // the components, entries [9*n_bins + bin] the traces.
    Eigen::VectorXd unscaled(const Eigen::VectorXd & x) const;

    // Raw delete-one-batch jackknife of unscaled() - use it for covariances (multiply entry i by the
    // binScale of its bin). Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeUnscaled() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return unscaled(x); }, acc);
    }

    // means (the estimator applied to the full-run means) and jackknife errors, binScale applied
    result normalised() const;

    // one line per bin: tau_centre hits, mean and error of G_00 ... G_22, mean and error of the trace;
    // preceded by '#' metadata lines
    void write(const std::string & path) const;
};

#endif // !GREEN_FUNC_HPP
