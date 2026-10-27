#ifndef GS_ENERGY_HPP
#define GS_ENERGY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include <simplemc/mpi/mpi_fwd.hpp>
#include <simplemc/serialize/concepts.hpp>
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
// Sign-free value (sign_free = true): E_sf = <X>, the plain mean over the sampled |W| ensemble. Since
// d ln|W|/d lambda = d ln W/d lambda, X is also the tau-scaling estimator of Z_|W| = sum |W|, so E_sf is the
// ground-state energy of the sign-free ensemble. Z_|W| >= |Z| gives E_sf <= E_0; they agree at first order (a
// one-line diagram is sign-definite) and <s>(tau_D) ~ e^{-(E_0 - E_sf) tau_D}. With <s> resolved at several
// tau_D, E_0 = E_sf - d ln<s>/d tau_D; when <s> is consistent with 0 the signed E is uncontrolled and tends to
// E_sf.
//
// Statistics: each measured sample is the vector [s X, s] (with sign_free: [s X, s, X]) in a simplemc batch
// accumulator; the energy is the delete-one-batch jackknife of the ratio of the first two means, E_sf that of
// the third mean (at least 2 full batches needed).
// Several ranks: collect the accumulator with simplemc_mpi_collect(comm, acc), and sum n_measured and
// n_negative, before calling normalised()/write().
struct gs_energy_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};
    const bool sign_free {false};       // also accumulate X without the sign (E_sf)

    acc_type acc;                       // one sample [s X, s] ([s X, s, X] with sign_free) per measured configuration
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_min)
    std::uint64_t n_negative {0};       // ... of which with sign -1

    struct result {
        long double energy {0.L};
        long double energy_error {0.L};
        long double energy_sf {0.L};            // sign-free energy; NaN without sign_free
        long double energy_sf_error {0.L};
    };

    gs_energy_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed = false,
                          std::size_t n_batches = 256, bool sign_free = false);

    // X for the current diagram, without the sign (see above); requires a finite, non-zero trace
    double estimate() const;

    void measure();

    // the energies as a function of the batch means x = [<s X>, <s>(, <X>)]: [E(, E_sf)]
    Eigen::VectorXd ratio(const Eigen::VectorXd & x) const {
        if (!sign_free) { return Eigen::VectorXd::Constant(1, x(0) / x(1)); }
        return Eigen::Vector2d(x(0) / x(1), x(2));
    }

    // Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeRatio() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return ratio(x); }, acc);
    }

    result normalised() const;

    // This rank's own result (before simplemc_mpi_collect merges the ranks), for the per-chain file: value and
    // jackknife error of each quantity in summaryNames() (E, and E_sf with sign_free), in that order; NaNs with
    // fewer than 2 full batches.
    std::vector<std::string> summaryNames() const;
    std::vector<double> summary() const;

    // '#' metadata lines, then one line: E err (with sign_free: E err E_sf err)
    void write(const std::string & path) const;

    // several ranks (collective): merges the batches of every rank into acc and sums the counters, on every rank
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, gs_energy_measurement & m);

    // checkpoint hooks (simplemc's persistent-state channel): the accumulator and the counters; the construction
    // parameters come from the input (main checks that the input still matches the checkpoint)
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const gs_energy_measurement & m){
        s.save_at("acc", m.acc);
        s.save_at("n_measured", m.n_measured);
        s.save_at("n_negative", m.n_negative);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S & s, gs_energy_measurement & m){
        s.load_at("acc", m.acc);
        s.load_at("n_measured", m.n_measured);
        s.load_at("n_negative", m.n_negative);
    }
};

#endif // !GS_ENERGY_HPP
