#ifndef QP_WEIGHTS_HPP
#define QP_WEIGHTS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include <simplemc/mpi/mpi_fwd.hpp>
#include "diagram/diagram_config.hpp"

// Quasiparticle residue and N-phonon weights of the polaron at total momentum p, from the trace over all
// boundary states.
//
// Every external line is a phonon present at tau = 0 = tau_D, so a diagram with N external lines is a
// contribution of the N-phonon states to tr_p e^{-tau_D H}. Summed over bands and electron momenta, the
// fraction of the trace in each sector is
//     Z_N = <s delta_{N_ext, N}> / <s>,      sum_N Z_N = 1,
// with s = sign(tr P), P = diagram_head->right_component (every other factor of the weight is positive).
// For tau_D >> 1/gap this is the phonon-number distribution of the ground state at p, averaged over its
// degenerate multiplet: Z_0 is the quasiparticle residue (the bare-electron component; N = 0 forces the
// electron momentum to p) and <N> = <s N_ext> / <s> the mean number of phonons in the cloud. No band or
// basis choice enters these.
//
// In the N = 0 sector the band matrix is kept as well, giving the band-resolved residue
//     R = <s delta_{N_ext, 0} U P U^T / |tr P|> / <s>,      tr R = Z_0,
// accumulated in the orbital (Gamma) basis, U = diagram_head->baseWF: the head's k equals p only up to
// the rounding of the external-line updates, and at p = 0 a residual k of order 1e-17 would give each
// sample an arbitrary band basis, while U P U^T does not depend on that basis (nor on the eigenvector
// signs). Each sample is symmetrised, (M + M^T)/2: the exact R is real symmetric, a single diagram's P is
// not. R is reported in the orbital basis and in the band basis at p, U(p)^T R U(p) (bands in the order
// diagonalizeLKHamiltonian returns them at p). At p = 0, R = (Z_0/3) I by cubic symmetry.
//
// tau_min / tau_fixed as in gs_energy_measurement: samples with tau_D < tau_min are skipped, and with
// tau_fixed every sample is used. e^{mu tau_D} is common to all sectors of a diagram, so it cancels in
// every ratio above and is not removed.
//
// Statistics: each measured sample is one sparse vector in a simplemc batch accumulator, laid out as
//     [0, n_max]            s delta_{N_ext, N}, N = 0..n_max (n_max = max_order_ext / 2: no overflow)
//     n_max + 1             s N_ext
//     n_max + 2 .. + 10     s delta_{N_ext, 0} (U P U^T / |tr P|)_ij, symmetrised, row-major
//     n_max + 11            s
// and every estimate is the delete-one-batch jackknife of a ratio to the last entry (at least 2 full
// batches needed). Several ranks: collect the accumulator with simplemc_mpi_collect(comm, acc), and sum
// hits, n_measured and n_negative, before calling normalised()/write().
struct qp_weights_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};
    const int n_max {0};                // largest number of external lines the diagram can hold
    const Eigen::Matrix3d U_p;          // band basis at p, for the band-resolved residue

    acc_type acc;                       // one sample per measured configuration
    std::vector<std::uint64_t> hits;    // measured samples per N (diagnostic)
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_min)
    std::uint64_t n_negative {0};       // ... of which with sign -1

    struct result {
        std::vector<long double> Z;     // Z_N, N = 0..n_max
        std::vector<long double> Z_error;
        long double mean_N {0.L};
        long double mean_N_error {0.L};
        Eigen::Matrix3d R_orb {Eigen::Matrix3d::Zero()};    // band-resolved residue, orbital basis
        Eigen::Matrix3d R_orb_error {Eigen::Matrix3d::Zero()};
        Eigen::Matrix3d R_band {Eigen::Matrix3d::Zero()};   // the same in the band basis at p
        Eigen::Matrix3d R_band_error {Eigen::Matrix3d::Zero()};
    };

    qp_weights_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed = false,
                           std::size_t n_batches = 256);

    void measure();

    // The estimates as a function of the batch means x: [Z_0..Z_{n_max}, <N>, R_orb (9, row-major),
    // R_band (9, row-major)].
    Eigen::VectorXd ratios(const Eigen::VectorXd & x) const;

    // Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeRatios() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return ratios(x); }, acc);
    }

    // means (the estimator applied to the full-run means) and jackknife errors
    result normalised() const;

    // '#' metadata lines with <N> and the two residue matrices, then one line per N: N hits Z_N err
    void write(const std::string & path) const;

private:
    int indexMeanN() const { return n_max + 1; }
    int indexResidue() const { return n_max + 2; }
    int indexSign() const { return n_max + 11; }

    // several ranks (collective): merges the batches of every rank into acc and sums the counters, on every rank
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, qp_weights_measurement & m);
};

#endif // !QP_WEIGHTS_HPP
