#ifndef GREEN_LAGUERRE_HPP
#define GREEN_LAGUERRE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include <simplemc/mpi/mpi_fwd.hpp>
#include <simplemc/serialize/concepts.hpp>
#include "diagram/diagram_config.hpp"

// Laguerre-coefficient estimator of the trace of the Green function, tr G(p, tau) (or, selecting phonons at
// the imaginary-time boundary, of the trace of the P function), with jackknife error bars: the analogue on
// [0, infinity) of the Legendre-coefficient measurement of Boehnke et al., PRB 84, 075145 (2011).
//
// The chain samples g(tau) = tr G(p, tau) e^{mu tau} (mu < 0) on (0, tau_max], which decays as
// e^{-(E_0 - mu) tau}. Instead of histogramming tau_D, every sample contributes to the coefficients
//     g_l = int_0^tau_max g(tau) phi_l(tau) dtau,   phi_l(tau) = sqrt(s) e^{-s tau/2} L_l(s tau),
// the Laguerre functions, orthonormal on [0, infinity) (measurement::laguerreFunctions). A sample at tau_D
// adds sign(tr P) phi_l(tau_D) to g_l, and |phi_l| <= sqrt(s): nothing large enters the statistics, and the
// sampling factor e^{mu tau_D} needs no removal - it is part of g, and comes back as e^{-mu tau} only when
// G is rebuilt. Normalisation as in green_trace_measurement, without the bin width:
//     g_l = Z_0 <sign phi_l(tau_D)> / <x_0>
// with x_0 the order-0 indicator (counted whatever the selection) and Z_0 its exact integral.
//
// Why Laguerre: a pole e^{-Delta tau} has the coefficients sqrt(s) (Delta - s/2)^l / (Delta + s/2)^{l+1}, a
// geometric sequence of ratio z = (Delta - s/2)/(Delta + s/2) in (-1, 1). The spectral sum g = sum_E A_E
// e^{-(E - mu) tau} therefore gives g_l = sum_E c_E z_E^l: smooth g, rapidly decaying g_l, and the noise
// sits in the high-l coefficients. Truncating at l_max is a noise filter; l_max is chosen from the plateau
// of the quantity of interest against l_max, as in the paper (the coefficients of every l <= l_max are
// written, so any smaller truncation can be applied afterwards).
//
// Scale s: scale > 0 is taken as given; scale <= 0 means s = 2 min_n (E_n(p) - mu), the order-0 decay rate
// (measurement::order0MinRate), for which a pure free tail is entirely in g_0. The polaron decays more
// slowly (E_0 < E_min(p)), so its poles sit at small z < 0.
//
// Truncation at tau_max: the expansion is of g times the indicator of (0, tau_max]. If g(tau_max) is not
// negligible, the step makes the coefficients decay slowly and the rebuilt G rings near tau_max: mu should
// satisfy (E_0 - mu) tau_max >> 1 (write() reports Delta_0 tau_max = min_n(E_n(p) - mu) tau_max as a guide).
//
// Rebuilt G: tr G(tau_j) = e^{-mu tau_j} sum_{l <= l_max} g_l phi_l(tau_j) at the n_grid centres of the
// green_trace bins, for a direct comparison. It is linear in the batch means, so the jackknife gives its
// errors with the correlations between coefficients; e^{-mu tau_j} is applied after the jackknife, in long
// double.
//
// Statistics: one dense vector sample per measure() of size l_max + 2 (coefficients, then the order-0
// indicator) in a simplemc batch accumulator; delete-one-batch jackknife; at least 2 full batches needed.
//
// Several ranks: collect with simplemc_mpi_collect before calling normalised()/write().
struct green_laguerre_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const int boundary_phonons {0};
    const int l_max {0};
    const double scale {0.};            // s, resolved (> 0)
    const int n_grid {0};
    const double tau_max {0.};

    acc_type acc;                       // one vector sample per measure(); count() = number of samples
    std::uint64_t n_selected {0};       // samples that entered the coefficients
    std::uint64_t n_negative {0};       // ... of which with sign -1
    std::uint64_t n_order0 {0};         // samples with no phonon line at all

    struct result {
        std::vector<long double> coeff;         // g_l, l = 0 .. l_max
        std::vector<long double> coeff_error;
        std::vector<long double> tau;           // grid: centres of n_grid bins over [0, tau_max]
        std::vector<long double> trace;         // rebuilt tr G(p, tau)
        std::vector<long double> trace_error;
    };

    green_laguerre_measurement(diagram_cfg * cfg, std::array<double, 3> p, int l_max, double scale, int n_grid,
                               int boundary_phonons = 0, std::size_t n_batches = 256);

    void measure();

    // The estimator before the e^{-mu tau_j} factors, as a function of the batch means x: entries [l] the
    // coefficients g_l, entries [l_max + 1 + j] the rebuilt g(tau_j).
    Eigen::VectorXd unscaled(const Eigen::VectorXd & x) const;

    // Raw delete-one-batch jackknife of unscaled(). Throws simplemc::simplemc_exception with fewer than 2
    // full batches.
    auto jackknifeUnscaled() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return unscaled(x); }, acc);
    }

    // means (the estimator applied to the full-run means) and jackknife errors, e^{-mu tau_j} applied
    result normalised() const;

    // '#' metadata lines, then two blocks separated by two blank lines (gnuplot index 0 and 1):
    // "l g_l err" and "tau trace err"
    void write(const std::string & path) const;

    // several ranks (collective): merges the batches of every rank into acc and sums the counters, on every rank
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, green_laguerre_measurement & m);

    // checkpoint hooks: the accumulator and the counters; the construction parameters come from the input
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const green_laguerre_measurement & m){
        s.save_at("acc", m.acc);
        s.save_at("n_selected", m.n_selected);
        s.save_at("n_negative", m.n_negative);
        s.save_at("n_order0", m.n_order0);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S & s, green_laguerre_measurement & m){
        s.load_at("acc", m.acc);
        s.load_at("n_selected", m.n_selected);
        s.load_at("n_negative", m.n_negative);
        s.load_at("n_order0", m.n_order0);
    }

private:
    std::vector<double> psi;            // scratch: e^{-x/2} L_l(x), l = 0 .. l_max
    std::vector<double> sample;         // scratch: one accumulator sample
    Eigen::MatrixXd basis;              // phi_l(tau_j), n_grid x (l_max + 1), for the rebuilt g
};

#endif // !GREEN_LAGUERRE_HPP
