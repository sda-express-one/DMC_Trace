#ifndef CONVERGENCE_MONITOR_HPP
#define CONVERGENCE_MONITOR_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <simplemc/accs/batch_acc.hpp>
#include "diagram/diagram_config.hpp"
#include "measurements/gs_energy.hpp"

// Thermalization measurement: once per simplemc cycle it records a few scalar summaries of the current diagram
// into an order-preserving batch accumulator, whose batch means are the draws of the R-hat convergence test
// (thermalization_control, measurement::diagnose).
//
// Scalars, in this order (names() gives them):
//     n_int       number of internal phonon lines
//     n_ext       number of external phonon lines
//     Y2_tau      |Y|^2 / tau_D, Y = sum_segments k dtau (the effective-mass fluctuation; |Y|^2 because <Y> = 0 by
//                 symmetry at p = 0 would make Y itself uninformative)
//     rms_w       root mean square phonon momentum over all lines (0 without lines)
//     rms_k       sqrt(sum_segments |k|^2 dtau / tau_D), the time-averaged electron momentum
//     tau_D       the diagram length              (Options::tau_D, for variable-length runs)
//     sign        sign(tr P) = +-1, the diagram's sign (Options::sign; a constant +1 without a sign problem, which
//                 diagnose() passes exactly: R-hat 1, ESS S)
//     energy      the energy estimate X of gs_energy_measurement (Options::energy; costlier: 3x3 products per segment)
//     M_xx M_yy M_zz   diagonal of U P U^T / |tr P|, the orbital fractions for C = 0 (Options::orbital)
// The sign of the diagram enters only through the sign scalar. Every call adds exactly one sample, so
// ranks running the same number of cycles hold the same number of batches.
struct convergence_monitor {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    struct Options {
        bool tau_D {false};
        bool sign {true};
        bool energy {false};
        bool orbital {false};
        std::size_t n_batches {1024};       // window: 256-512 draws per split chain, enough for a total ESS of 400 on 1 rank
    };

    diagram_cfg * cfg {nullptr};
    Options opt;
    std::vector<std::string> scalar_names;
    acc_type acc;
    std::shared_ptr<gs_energy_measurement> energy_estimator;    // only with Options::energy (shared: the monitor stays copyable)

    explicit convergence_monitor(diagram_cfg * cfg, Options opt);
    convergence_monitor(diagram_cfg * cfg) : convergence_monitor(cfg, Options{}) {}

    const std::vector<std::string> & names() const { return scalar_names; }
    std::size_t size() const { return scalar_names.size(); }

    // the current diagram's scalars (also what measure() accumulates)
    std::vector<double> scalars() const;

    void measure();

    // The draws for the convergence test: the second half of the full batches (the first half is discarded as
    // warm-up, so the window grows with the run), as means, scalar-major: out[k * B + b] for scalar k, batch b.
    // B is returned through `batches`; it is 0 before the accumulator's first merge.
    std::vector<double> window(std::size_t & batches) const;

    // the largest B window() can return: the accumulator holds fewer than 2 n_batches full batches
    std::size_t maxWindowBatches() const { return opt.n_batches; }
};

#endif // !CONVERGENCE_MONITOR_HPP
