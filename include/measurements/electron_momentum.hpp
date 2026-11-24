#ifndef ELECTRON_MOMENTUM_HPP
#define ELECTRON_MOMENTUM_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <mpi.h>
#include <simplemc/accs/batch_acc.hpp>
#include <simplemc/accs/jackknife.hpp>
#include <simplemc/mpi/mpi_fwd.hpp>
#include <simplemc/serialize/concepts.hpp>
#include "diagram/diagram_config.hpp"

// Electron momentum distribution n(k) of the polaron at total momentum p, time-averaged over the diagram and
// resolved by the number N of phonons present.
//
// The trace tr_p e^{-tau_D H} is cyclic: every time slice tau' of a diagram is equivalent to the boundary. At a slice
// inside segment i the electron has the segment's momentum k_i and the phonons are the lines crossing the slice, N_i
// of them. Inserting the scalar operator delta(k^ - k) 1_orb delta(N^ - N) at tau' therefore multiplies the diagram's
// weight by an indicator, and averaging tau' over [0, tau_D) gives the per-sample estimator
//     n_N(k) = < s sum_i (dtau_i / tau_D) delta_{N_i, N} delta(k_i - k) > / <s>,
// with s = sign(tr P). It is the same quantity as the boundary-only histogram of k' = p - sum_ext w in the sector of
// N_ext = N lines, but every slice of the diagram contributes. Summed over k it gives the time-averaged phonon-number
// distribution Z_N^t (sum_N Z_N^t = 1), which equals qp_weights' Z_N at large tau_D. N_i comes from one walk along
// the diagram: N = N_ext on the head segment (the external lines cross the boundary), +1 at every creation vertex
// (type > 0) and -1 at every annihilation vertex (type < 0).
//
// The N = 0 sector is the coherent part Z_0 delta(k - p) (k_i = p exactly when no line crosses), so its histogram is
// not kept: only its weight Z_0^t. The outputs are
//   per sector N = 0 .. n_max (the last one holds every N >= n_max):
//       Z_N^t, <k_a>_N and <k_a k_b>_N (the weight-normalised moments are these divided by Z_N^t), and a radial
//       histogram of r = |k - p| on the bounded map u = r / (r + k_s) (n_bins_r bins over [0, 1), the last bin
//       reaching r = infinity: its weight only);
//   summed over N >= 1: a 3D histogram of k - p on the per-axis bounded map u_a = q_a / (|q_a| + k_s) (n_bins_3d bins
//       over (-1, 1) per axis, the outermost cells reaching +-infinity).
// The scale k_s is fixed at construction (in diagramc from the thermalization, see main.cpp); it sets only where the
// bins are fine, no sample ever falls outside a map. Every estimate is signed (a ratio to <s>); there is no sign-free
// variant. No symmetry is imposed: the invariance of n(k) under the little group of p is left as a check.
//
// Statistics: the scalars and the radial histograms are one sparse sample per configuration in `acc` (n_batches),
// with the delete-one-batch jackknife (Welford) of simplemc. The 3D histogram has its own accumulator `acc3d`
// (n_batches_3d batches, n_bins_3d^3 + 1 entries: the cells and s); its jackknife is done here per cell (two-pass on
// the batch means), since simplemc's jackknife would build the full covariance matrix of the cells.
// Several ranks: simplemc_mpi_collect(comm, m) before normalised()/write(). It gathers `acc` as usual, but never
// acc3d: simplemc's collect would copy every rank's batches to every rank (memory ~ ranks x n_batches_3d x cells).
// Instead the 3D jackknife runs distributed over the ranks' own batches (equal batch counts enforced as in simplemc's
// gather, by merging), with vector all-reduces only, and its result is kept for normalised3d()/write3d().
struct electron_momentum_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;
    using size_type = acc_type::size_type;

    // 1D bounded map: u = x / (|x| + scale) on (-1, 1) (symmetric) or u = x / (x + scale) on [0, 1) (x >= 0), with
    // n equal bins in u; bin edges in x are scale u / (1 - |u|) (+-infinity at u = +-1).
    struct bounded_map {
        double scale {1.};
        int n {2};
        bool symmetric {true};

        int index(double x) const;
        double edge(int i) const;                   // i = 0 .. n
        double center(int i) const;                 // the x of the bin's middle u (finite for every bin)
    };

    diagram_cfg * const cfg {nullptr};
    const std::array<double, 3> p;      // external momentum: the k_init the diagram was built with
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};
    const int n_max {8};                // sectors N = 0 .. n_max, the last one N >= n_max
    const int n_bins_r {64};
    const int n_bins_3d {32};
    double k_scale {1.};                // k_s: restored from a checkpoint (the bin layout does not depend on it)

    acc_type acc;                       // per sector [Z, <k_a> (3), <k_a k_b> (6), radial (n_bins_r)], then s
    acc_type acc3d;                     // n_bins_3d^3 cells (x slowest), then s
    std::uint64_t n_measured {0};
    std::uint64_t n_negative {0};

    struct result {
        std::vector<double> Z, Z_error;                         // Z_N^t, N = 0..n_max
        std::vector<std::array<double, 3>> k, k_error;          // <k_a delta_{N_i, N}> (not divided by Z_N)
        std::vector<std::array<double, 6>> kk, kk_error;        // <k_a k_b delta_{N_i, N}>: xx yy zz xy xz yz
        std::vector<std::vector<double>> radial, radial_error;  // [N][bin] weights
    };

    struct result3d {
        std::vector<double> weight, error;                      // n_bins_3d^3 cells, index (ix n + iy) n + iz
        std::size_t batches {0};                                // batches over all ranks used by the jackknife
        std::uint64_t batch_count {0};                          // samples per batch
    };
    std::optional<result3d> collected3d;                        // set by simplemc_mpi_collect

    electron_momentum_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed,
                                  std::size_t n_batches, double k_scale, int n_bins_3d = 32, int n_bins_r = 64,
                                  int n_max = 8, std::size_t n_batches_3d = 64);

    void measure();

    bounded_map radialMap() const { return {k_scale, n_bins_r, false}; }
    bounded_map axisMap() const { return {k_scale, n_bins_3d, true}; }

    // per-sample pieces (also what measure() accumulates), for tests: (index, value) pairs of acc and of acc3d,
    // sorted by index, duplicates merged, the sign included; and the sector of every segment, in time order
    void sample(std::vector<std::pair<size_type, double>> & out, std::vector<std::pair<size_type, double>> & out3d) const;
    std::vector<int> segmentSectors() const;

    // the estimates as a function of the batch means x of acc: every entry divided by <s>
    Eigen::VectorXd ratios(const Eigen::VectorXd & x) const { return x.head(x.size() - 1) / x(x.size() - 1); }

    // Throw simplemc::simplemc_exception (or std::runtime_error for the 3D part) with fewer than 2 full batches.
    // normalised3d(): the collected result after simplemc_mpi_collect, else this rank's own batches.
    result normalised() const;
    result3d normalised3d() const { return collected3d ? *collected3d : jackknife3d(MPI_COMM_NULL); }

    // the per-cell delete-one-batch jackknife of acc3d (two-pass); with a communicator, over the batches of every rank
    // (collective), with MPI_COMM_NULL over this rank's only
    result3d jackknife3d(MPI_Comm comm) const;

    // <prefix>_electron_momentum.dat: '#' metadata, the per-sector scalars, then the radial table; also writes the 3D
    // histogram next to it (write3d, path with ".dat" replaced by "_3d.dat")
    void write(const std::string & path) const;
    // <prefix>_electron_momentum_3d.dat: '#' metadata and the per-axis edges, then one line per cell
    void write3d(const std::string & path) const;

    int sectorSize() const { return 10 + n_bins_r; }
    int indexSector(int N) const { return N * sectorSize(); }
    int indexSign() const { return (n_max + 1) * sectorSize(); }
    int nCells() const { return n_bins_3d * n_bins_3d * n_bins_3d; }

    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, electron_momentum_measurement & m);

    // checkpoint hooks: the accumulators, the counters and k_s (the bin layout comes from the input)
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const electron_momentum_measurement & m){
        s.save_at("acc", m.acc);
        s.save_at("acc3d", m.acc3d);
        s.save_at("k_scale", m.k_scale);
        s.save_at("n_measured", m.n_measured);
        s.save_at("n_negative", m.n_negative);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S & s, electron_momentum_measurement & m){
        acc_type a, a3;
        s.load_at("acc", a);
        s.load_at("acc3d", a3);
        if (a.size() != m.acc.size() || a3.size() != m.acc3d.size()) {
            // an inactive estimator is built with the smallest layout (main.cpp): nothing to restore from it
            if (a.count() == 0 && a3.count() == 0) { return; }
            throw std::runtime_error("checkpoint: electron_momentum accumulators of a different size (input changed?)");
        }
        m.acc = std::move(a);
        m.acc3d = std::move(a3);
        s.load_at("k_scale", m.k_scale);
        s.load_at("n_measured", m.n_measured);
        s.load_at("n_negative", m.n_negative);
    }
};

#endif // !ELECTRON_MOMENTUM_HPP
