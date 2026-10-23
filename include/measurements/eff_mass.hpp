#ifndef EFF_MASS_HPP
#define EFF_MASS_HPP

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

// Polaron effective mass and group velocity at total momentum p, from the sampling at p alone. Valid
// ONLY for the isotropic Froehlich limit of the LK model, A = B and C = 0 (check with applicable()).
//
// There the three bands are identical parabolas, E(k) = k^2/2m with m = 1/(2A): in the orbital basis every
// electron propagator is proportional to the identity and the closed product of overlaps is the identity,
// so tr P = 3 exp(-sum_i |k_i|^2 dtau_i / 2m). Shifting the total momentum p -> p + d shifts every
// segment's momentum by d (external lines included), and with Y = sum_segments k dtau the weight changes as
//     ln W(p + d) - ln W(p) = -(d^2 tau + 2 d.Y) / 2m,
// couplings and phonon propagators being independent of p. Comparing ln G(p + d, tau) - ln G(p, tau) with
// -tau (E(p + d) - E(p)) to second order in d gives, for tau >> 1/gap,
//     group velocity      v_a = <Y_a / tau> / m
//     inverse-mass tensor (m/m*)_ab = delta_ab - [<Y_a Y_b> - <Y_a><Y_b>] / (m tau)
// (Mishchenko et al., PRB 62, 6317 (2000), for the single band). At p = 0, <Y> = 0 and the tensor is
// isotropic by symmetry. With C = 0 and A != B the bands are uniaxial and the estimator needs the orbital
// fractions of each sample instead (markdown_notes/lk_bands_implementation.md, section 8.3).
//
// Per sample, with s = sign(tr P), the accumulated vector is
//     [0..2]   s Y_a / tau
//     [3..8]   s Y_a Y_b / tau     (xx, yy, zz, xy, xz, yz)
//     [9]      s
// and the covariance term is C_ab = Cov(Y_a, Y_b) / tau:
//   - tau_fixed (every sample has the same length tau_D, read from the diagram at construction):
//         C_ab = <Y_a Y_b / tau> - tau_D <Y_a / tau><Y_b / tau>,
//     exact at any p;
//   - varying tau_D: C_ab = <Y_a Y_b / tau>, without the <Y><Y> subtraction. Correct only at p = 0, where
//     <Y> = 0 by symmetry; at p != 0 samples of different lengths cannot be combined into a covariance
//     (the spread of <Y>/sqrt(tau) ~ m v sqrt(tau) over the lengths would enter it), so use tau_fixed there.
//     The velocity <Y/tau>/m is valid either way.
//
// Finite tau: <Y_a Y_b> - <Y_a><Y_b> = a_ab + b_ab tau, so the aggregate carries an O(1/tau) bias a/(m tau).
// Two fixed-length runs remove it: (m/m*)_ab = delta_ab - (tau_2 C_ab(tau_2) - tau_1 C_ab(tau_1)) /
// (m (tau_2 - tau_1)), which is why C is reported as well.
//
// tau_min / tau_fixed as in gs_energy_measurement. e^{mu tau_D} is a sampling device only and, at fixed
// tau_D, a constant; with varying tau_D it reweights the lengths, which is harmless for converged samples.
// Statistics and several ranks as in qp_weights_measurement: one sample per measured configuration in a
// simplemc batch accumulator, delete-one-batch jackknife of the ratios to <s>.
struct eff_mass_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};
    const double m {0.};                // bare band mass 1/(2A), fixed at construction
    const double tau_length {-1.};      // the fixed tau_D when tau_fixed (from the diagram at construction), else -1

    acc_type acc;                       // one sample per measured configuration
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_min)
    std::uint64_t n_negative {0};       // ... of which with sign -1

    struct result {
        Eigen::Matrix3d inv_mass {Eigen::Matrix3d::Zero()};        // (m/m*)_ab
        Eigen::Matrix3d inv_mass_error {Eigen::Matrix3d::Zero()};
        long double inv_mass_iso {0.L};                            // tr(m/m*) / 3
        long double inv_mass_iso_error {0.L};
        long double mass_ratio {0.L};                              // m*/m = 3 / tr(m/m*)
        long double mass_ratio_error {0.L};
        Eigen::Vector3d velocity {Eigen::Vector3d::Zero()};        // group velocity v_a
        Eigen::Vector3d velocity_error {Eigen::Vector3d::Zero()};
        Eigen::Matrix3d C {Eigen::Matrix3d::Zero()};               // Cov(Y_a, Y_b) / tau, for the slope recipe
        Eigen::Matrix3d C_error {Eigen::Matrix3d::Zero()};
    };

    // true iff the LK parameters set in weight::LKMatrix describe the isotropic Froehlich limit
    static bool applicable();

    eff_mass_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed = false,
                         std::size_t n_batches = 256);

    // Y = sum over segments of k dtau, for the current diagram
    Eigen::Vector3d pathMomentum() const;

    void measure();

    // The estimates as a function of the batch means x: [inv_mass (9, row-major), inv_mass_iso,
    // mass_ratio, velocity (3), C (9, row-major)].
    Eigen::VectorXd ratios(const Eigen::VectorXd & x) const;

    // Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeRatios() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return ratios(x); }, acc);
    }

    // means (the estimator applied to the full-run means) and jackknife errors
    result normalised() const;

    // This rank's own result (before simplemc_mpi_collect merges the ranks), for the per-chain file: value and
    // jackknife error of each quantity in summaryNames(), in that order; NaNs with fewer than 2 full batches.
    static std::vector<std::string> summaryNames();
    std::vector<double> summary() const;

    // '#' metadata lines, then the scalar m/m* and m*/m, the tensor, v and C
    void write(const std::string & path) const;

    // several ranks (collective): merges the batches of every rank into acc and sums the counters, on every rank
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, eff_mass_measurement & m);

    // checkpoint hooks (simplemc's persistent-state channel): the accumulator and the counters; the construction
    // parameters come from the input (main checks that the input still matches the checkpoint)
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const eff_mass_measurement & m){
        s.save_at("acc", m.acc);
        s.save_at("n_measured", m.n_measured);
        s.save_at("n_negative", m.n_negative);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S & s, eff_mass_measurement & m){
        s.load_at("acc", m.acc);
        s.load_at("n_measured", m.n_measured);
        s.load_at("n_negative", m.n_negative);
    }
};

#endif // !EFF_MASS_HPP
