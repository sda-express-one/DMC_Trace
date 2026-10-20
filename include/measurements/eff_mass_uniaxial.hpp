#ifndef EFF_MASS_UNIAXIAL_HPP
#define EFF_MASS_UNIAXIAL_HPP

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

// Polaron effective masses of the uniaxial LK model, C = 0 (check with applicable(); A = B is included and
// reduces to eff_mass_measurement's isotropic case).
//
// With C = 0, H_LK is diagonal in the orbital basis (x, y, z): orbital i is a uniaxial band
//     e_i(k) = sum_a k_a^2 / (2 m_ia),   m_ia = m_par = 1/(2A) if a = i, else m_perp = 1/(2B),
// and the vertex overlaps are signed permutations, the identity in the orbital basis, so a vertex never
// changes the orbital. The trace therefore splits exactly into three independent single-band sectors,
//     tr P = W_x + W_y + W_z,   W_i ~ e^{-B K} e^{-(A - B) K_i},   K_i = sum_segments k_i^2 dtau,
// sharing every phonon factor. An unweighted average over the chain (which samples sum_i W_i) mixes the
// sectors - it gives only the trace combination of the masses. Each sector's own average follows by
// reweighting with the per-sample orbital fraction w_i = W_i / sum_j W_j, read from the diagram as the
// diagonal of M = U P U^T / |tr P| (U = diagram_head->baseWF, so M_ii = s w_i; M is diagonal for C = 0):
//     <O>_i = <M_ii O> / <M_ii>.
// Within sector i the weight factorises per axis, ln W_i(p + d) - ln W_i(p) = -sum_a (d_a^2 tau + 2 d_a Y_a)
// / (2 m_ia), with Y = sum_segments k dtau (measurement::pathMomentum), so to second order in d, for
// tau >> 1/gap:
//     inverse-mass tensor  (1/m*)_i,ab = delta_ab / m_ia - Cov_i(Y_a, Y_b) / (m_ia m_ib tau)
//     group velocity       v_i,a = <Y_a / tau>_i / m_ia
// At p = 0 the three sectors are equivalent by cubic symmetry, and are pooled (weighted by <w_i>) into the
// longitudinal (a = i) and transverse (a != i) masses m*_par and m*_perp. At p != 0 the sectors have
// different energies and the higher ones' weights decay with tau_D: use the per-orbital tensors there, and
// only with tau_fixed (as in eff_mass_measurement, varying tau_D cannot form a covariance at p != 0).
//
// Per sample, with s = sign(tr P), the accumulated vector holds one block of 10 per orbital i:
//     [10 i]              M_ii
//     [10 i + 1 .. + 3]   M_ii Y_a / tau
//     [10 i + 4 .. + 9]   M_ii Y_a Y_b / tau     (xx, yy, zz, xy, xz, yz)
// and [30] s. The covariance is C_i,ab = <Y_a Y_b / tau>_i - subtract <Y_a / tau>_i <Y_b / tau>_i, with
// subtract = tau_D for tau_fixed (exact at any p) and 0 for varying tau_D (valid at p = 0 only), exactly as
// in eff_mass_measurement. tau_length is read from the diagram at construction, so the diagram must already
// have its final length then; measure() asserts it does not change.
//
// Finite tau: the aggregate carries an O(1/tau) bias; two fixed lengths remove it, per orbital,
// (1/m*)_i,ab = delta_ab/m_ia - (tau_2 C_i,ab(tau_2) - tau_1 C_i,ab(tau_1)) / (m_ia m_ib (tau_2 - tau_1)),
// which is why the C_i are reported.
//
// tau_min / tau_fixed, statistics and several ranks as in eff_mass_measurement.
struct eff_mass_uniaxial_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};
    const double m_par {0.};            // bare mass along an orbital's own axis, 1/(2A)
    const double m_perp {0.};           // bare mass across it, 1/(2B)
    const double tau_length {-1.};      // the fixed tau_D when tau_fixed (from the diagram at construction), else -1

    acc_type acc;                       // one sample per measured configuration
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_min)
    std::uint64_t n_negative {0};       // ... of which with sign -1

    struct result {
        Eigen::Vector3d w {Eigen::Vector3d::Zero()};                // <w_i>, orbital fractions
        Eigen::Vector3d w_error {Eigen::Vector3d::Zero()};
        std::array<Eigen::Matrix3d, 3> inv_mass {};                 // (1/m*)_i,ab in a.u., one tensor per orbital
        std::array<Eigen::Matrix3d, 3> inv_mass_error {};
        Eigen::Matrix3d velocity {Eigen::Matrix3d::Zero()};         // row i: v_i,a
        Eigen::Matrix3d velocity_error {Eigen::Matrix3d::Zero()};
        long double inv_mass_par {0.L};                             // pooled 1/m*_par (a.u.)
        long double inv_mass_par_error {0.L};
        long double inv_mass_perp {0.L};                            // pooled 1/m*_perp (a.u.)
        long double inv_mass_perp_error {0.L};
        long double mass_ratio_par {0.L};                           // m*_par / m_par
        long double mass_ratio_par_error {0.L};
        long double mass_ratio_perp {0.L};                          // m*_perp / m_perp
        long double mass_ratio_perp_error {0.L};
        std::array<Eigen::Matrix3d, 3> C {};                        // Cov_i(Y_a, Y_b) / tau, for the slope recipe
        std::array<Eigen::Matrix3d, 3> C_error {};
    };

    // true iff the LK parameters set in weight::LKMatrix have C = 0
    static bool applicable();

    eff_mass_uniaxial_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed = false,
                                  std::size_t n_batches = 256);

    // bare mass of orbital i along axis a
    double bareMass(int i, int a) const { return i == a ? m_par : m_perp; }

    void measure();

    // The estimates as a function of the batch means x:
    //     [0..2]    <w_i>
    //     [3..29]   (1/m*)_i,ab        index 3 + 9 i + 3 a + b
    //     [30..38]  v_i,a              index 30 + 3 i + a
    //     [39..42]  1/m*_par, 1/m*_perp, m*_par/m_par, m*_perp/m_perp
    //     [43..69]  C_i,ab             index 43 + 9 i + 3 a + b
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

    // '#' metadata lines with the orbital fractions, per-orbital tensors, velocities and C, then one line with
    // the pooled masses
    void write(const std::string & path) const;

    // several ranks (collective): merges the batches of every rank into acc and sums the counters, on every rank
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, eff_mass_uniaxial_measurement & m);
};

#endif // !EFF_MASS_UNIAXIAL_HPP
