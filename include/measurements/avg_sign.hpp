#ifndef AVG_SIGN_HPP
#define AVG_SIGN_HPP

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

// Average sign of the sampled diagrams, <s> = sum W / sum |W|.
//
// The chain samples |W| and every factor of W except tr P is positive, so s = sign(tr P) (P =
// diagram_head->right_component) and <s> is the plain mean of s over the samples. With tau_fixed = true every
// diagram has the same length tau_D, and <s> reduces to the ratio of the signed to the absolute trace sum at that
// length - what green_trace (split_sign) gives for the single bin containing tau_D. With variable length it is the
// overall <s> of the diagrams with tau_D >= tau_min, the same selection gs_energy uses (the tau profile is
// green_trace's split_sign). For the measured samples <s> = 1 - 2 n_negative / n_measured exactly.
//
// <s> sets the cost of every signed estimator: their relative errors grow like 1/<s>. It is ~1 without a sign
// problem (C = 0: exactly 1) and drops with tau_D and the number of vertices for degenerate bands (C != 0).
//
// Statistics: each sample is the 1-vector [s] in a simplemc batch accumulator, the error the delete-one-batch
// jackknife of its mean (at least 2 full batches). Several ranks: collect with simplemc_mpi_collect(comm, m), which
// merges the batches and sums n_measured and n_negative, before normalised()/write().
struct avg_sign_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const double tau_min {0.};          // -1 (unused) when tau_fixed
    const bool tau_fixed {false};

    acc_type acc;                       // one sample [s] per measured configuration
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_min)
    std::uint64_t n_negative {0};       // ... of which with sign -1

    struct result {
        long double sign {0.L};
        long double sign_error {0.L};
    };

    avg_sign_measurement(diagram_cfg * cfg, double tau_min, bool tau_fixed = false, std::size_t n_batches = 256);

    void measure();

    // Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeMean() const {
        return simplemc::jackknife([](const Eigen::VectorXd & x){ return Eigen::VectorXd(x); }, acc);
    }

    result normalised() const;

    // This rank's own result, for the per-chain file: value and jackknife error of <s>; NaNs with fewer than 2 full
    // batches.
    static std::vector<std::string> summaryNames();
    std::vector<double> summary() const;

    // '#' metadata lines (counts, negative fraction), then one line: sign err
    void write(const std::string & path) const;

    // several ranks (collective): merges the batches of every rank into acc and sums the counters, on every rank
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, avg_sign_measurement & m);

    // checkpoint hooks (simplemc's persistent-state channel): the accumulator and the counters; the construction
    // parameters come from the input (main checks that the input still matches the checkpoint)
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const avg_sign_measurement & m){
        s.save_at("acc", m.acc);
        s.save_at("n_measured", m.n_measured);
        s.save_at("n_negative", m.n_negative);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S & s, avg_sign_measurement & m){
        s.load_at("acc", m.acc);
        s.load_at("n_measured", m.n_measured);
        s.load_at("n_negative", m.n_negative);
    }
};

#endif // !AVG_SIGN_HPP
