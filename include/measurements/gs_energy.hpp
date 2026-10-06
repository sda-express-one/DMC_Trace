#ifndef GS_ENERGY_HPP
#define GS_ENERGY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <Eigen/Core>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include "diagram/diagram_config.hpp"

// Ground-state energy from the tau-scaling ("exact") estimator of E(tau) = -d ln G/dtau, which tends
// to E_0(p) for tau >> 1/omega.
//
// Scaling every time of a diagram of length tau by lambda multiplies its measure by lambda^n (n = number
// of vertices, each an integrated time), every phonon propagator e^{-omega l} by e^{-omega l (lambda-1)}
// and every electron action A_i = e^{-E_i dtau_i} likewise, so tau dG/dtau = sum over diagrams of
// W * (n + d ln W/d lambda). Per sample this gives
//     X = ( -n + sum_lines omega l + sum_segments dtau_i tr(L_i wf_i E_i A_i wf_{i+1} R_{i+1}) / tr P ) / tau
// with E_i = diag(k_i^2/2m) the segment's band energies (no mu: e^{mu tau} is a sampling device only),
// l = tau2 - tau1 for an internal line and tau - tau_cre + tau_ann for an external (wrapped) one, and
// P = diagram_head->right_component. Couplings and phonon propagators are common to the trace with and
// without the inserted E_i, so they cancel in that ratio. With s = sign(tr P), E = <s X> / <s>.
//
// tau_min: samples with tau_D < tau_min are skipped (E(tau) has not converged to E_0 there). With
// tau_fixed = true every sample is used (runs in which tau_D is held at a large fixed value).
// Every boundary sector is included, i.e. the estimator tracks the lowest state of the full trace.
//
// Statistics: each measured sample is the vector [s X, s] in a simplemc batch accumulator; the energy
// is the delete-one-batch jackknife of the ratio of the two means (at least 2 full batches needed).
// Several ranks: collect the accumulator with simplemc_mpi_collect(comm, acc), and sum n_measured,
// before calling normalised()/write().
struct gs_energy_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};

    acc_type acc;                       // one sample [s X, s] per measured configuration
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_min)

    struct result {
        long double energy {0.L};
        long double energy_error {0.L};
    };

    gs_energy_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed = false,
                          std::size_t n_batches = 256);

    // X for the current diagram, without the sign (see above); requires a finite, non-zero trace
    double estimate() const;

    void measure();

    // the energy as a function of the batch means x = [<s X>, <s>]
    Eigen::VectorXd ratio(const Eigen::VectorXd & x) const {
        return Eigen::VectorXd::Constant(1, x(0) / x(1));
    }

    // Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeRatio() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return ratio(x); }, acc);
    }

    result normalised() const;

    // '#' metadata lines, then one line: E err
    void write(const std::string & path) const;
};

#endif // !GS_ENERGY_HPP
