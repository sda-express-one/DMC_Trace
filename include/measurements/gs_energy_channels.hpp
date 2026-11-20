#ifndef GS_ENERGY_CHANNELS_HPP
#define GS_ENERGY_CHANNELS_HPP

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
#include "comp_method/little_group.hpp"
#include "diagram/diagram_config.hpp"

// Ground-state energy of every symmetry channel of p: the tau-scaling estimator of gs_energy_measurement applied to
// the projected vacuum-sector Green function tr(Q_c G(p, tau)) instead of the trace
// (latex_notes/symmetry_channel_excited_band.tex, Sec. 6; validated against green_func in runs/alas3_channel100).
//
// Channels (comp_method/little_group.hpp): the isotypic components of the little group G_p of p on the orbital space,
// found automatically. Along [100] they are x (the longitudinal band, A p^2) and (y, z) (the transverse pair, B p^2);
// along [110] x+y, x-y, z; along [111] x+y+z and its orthogonal plane. Each channel's lowest state is a ground state of
// its own symmetry sector, so excited bands of the trace become accessible, with no multi-exponential fit, as long as
// they lie below the one-phonon threshold E_0(0) + omega. A generic p, or p = 0, has a single channel: the vacuum trace.
//
// Per vacuum sample (no external line), with U = diagram_head->baseWF (eigenvectors at k_end = p), P the diagram's band
// matrix and Qt_c = U^T Q_c U the projector in the head's band basis:
//     r_c = tr(Qt_c P) / |tr P|                                  (sum_c r_c = sign(tr P))
//     Y_c = [ (-n + sum_lines omega l) tr(Qt_c P) + sum_i dtau_i tr(Qt_c S_i) ] / (tau |tr P|)
//                                                                (sum_c Y_c = sign(tr P) X of gs_energy)
// with S_i the segment products of measurements/tau_scaling.hpp, and E_c = <Y_c> / <r_c>. Nothing is divided by the
// projected trace, which can vanish for a single diagram.
//
// Only the vacuum sector: with external lines the orbital index of the electron at k_end = p - sum(w) does not project
// the boundary state onto a representation of G_p (note, Sec. 5), so such samples enter as zeros (they still count
// for the share). gs_energy itself is unaffected and keeps every sector.
//
// tau_min / tau_fixed as in gs_energy: variable-length runs use tau_D >= tau_min only, which must be long enough for
// every channel (tau_min >> 1 / (distance of the channel's lowest state to the next state of the same channel)).
//
// Statistics: one sample [Y_1, r_1, ..., Y_nc, r_nc, v] per measured configuration (v = 1 for the vacuum sector) in a
// simplemc batch accumulator; delete-one-batch jackknife (Welford) of E_c and of the shares <r_c> and <v>.
// Output: <prefix>_gs_energy_channels.dat (write()) and, through summary(), <prefix>_gs_energy_channels_chains.dat.
struct gs_energy_channels_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};
    const int group_order {0};          // |G_p|
    const std::vector<symmetry::Channel> channels;

    acc_type acc;
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_min, non-zero trace)
    std::uint64_t n_vacuum {0};         // ... of which in the vacuum sector
    std::uint64_t n_negative {0};       // ... vacuum samples with sign -1

    struct result {
        std::vector<long double> energy, energy_error;  // per channel
        std::vector<long double> share, share_error;    // <r_c>: the channel's average sign per measured sample
        long double vacuum {0.L}, vacuum_error {0.L};   // fraction of measured samples in the vacuum sector
    };

    gs_energy_channels_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed = false,
                                   std::size_t n_batches = 256);

    int nChannels() const { return static_cast<int>(channels.size()); }

    // [Y_1, r_1, ..., Y_nc, r_nc] of the current diagram; requires the vacuum sector and a finite, non-zero trace
    Eigen::VectorXd estimate() const;

    void measure();

    // as a function of the batch means: [E_1 .. E_nc, share_1 .. share_nc, vacuum]
    Eigen::VectorXd ratio(const Eigen::VectorXd & x) const;

    // Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeRatio() const {
        return simplemc::jackknife<simplemc::varalg::welford>([this](const Eigen::VectorXd & x){ return ratio(x); }, acc);
    }

    result normalised() const;

    // per-chain file (main's writeChains): E_ch0 .. E_ch<nc-1>, value and error each; NaNs before 2 full batches
    std::vector<std::string> summaryNames() const;
    std::vector<double> summary() const;

    // '#' metadata (p, |G_p|, each channel's dimension, basis and bare lambda), then one row per channel:
    // "channel dim E err share err"
    void write(const std::string & path) const;

    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, gs_energy_channels_measurement & m);

    template <simplemc::serializer S>
    friend void simplemc_save(S s, const gs_energy_channels_measurement & m){
        s.save_at("acc", m.acc);
        s.save_at("n_measured", m.n_measured);
        s.save_at("n_vacuum", m.n_vacuum);
        s.save_at("n_negative", m.n_negative);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S & s, gs_energy_channels_measurement & m){
        s.load_at("acc", m.acc);
        s.load_at("n_measured", m.n_measured);
        s.load_at("n_vacuum", m.n_vacuum);
        s.load_at("n_negative", m.n_negative);
    }
};

#endif // !GS_ENERGY_CHANNELS_HPP
