#ifndef STATISTICS_HPP
#define STATISTICS_HPP

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

// Diagram statistics: the average number of internal and external phonon lines, n_int and n_ext (each manager's
// current_length / 2, as convergence_monitor's scalars), and their sum n_total.
//
// Only diagrams with tau_D >= tau_cutoff are collected (0: all). Two averages of each quantity are reported:
//   plain    the mean over the sampled ensemble, which is the sign-free one (the chain samples |W|): <n>;
//   signed   the expectation value in the signed ensemble, <s n> / <s> (s = sign(tr P), as avg_sign) - the physical
//            one; it equals plain without a sign problem (C = 0) and is as noisy as 1/<s> otherwise.
// With split_sign also the plain means over the positive and over the negative diagrams, n_plus = <n I+> / <I+> and
// n_minus = <n I-> / <I->, and the fraction of positive diagrams f+ = <I+>; then
//   plain = f+ n_plus + (1 - f+) n_minus,   signed = (f+ n_plus - (1 - f+) n_minus) / (2 f+ - 1).
// Without negative diagrams n_minus is 0/0: NaN, written as such.
//
// Statistics: each measured diagram is the sample [n_int, n_ext, s, s n_int, s n_ext] (with split_sign also
// [I+, I+ n_int, I+ n_ext]; the negative ones follow as total minus positive) in a simplemc batch accumulator; every
// quantity is the delete-one-batch jackknife of its function of the batch means (at least 2 full batches).
// Several ranks: collect with simplemc_mpi_collect(comm, m) before normalised()/write().
struct statistics_measurement {
    using acc_type = simplemc::batch_acc_dynamic<double>;

    diagram_cfg * const cfg {nullptr};
    const double tau_cutoff {0.};
    const bool split_sign {false};

    acc_type acc;                       // one sample per measured configuration (see above)
    std::uint64_t n_measured {0};       // configurations accumulated (tau_D >= tau_cutoff)
    std::uint64_t n_negative {0};       // ... of which with sign -1

    // values and jackknife errors, in the order of summaryNames()
    struct result {
        std::vector<double> value;
        std::vector<double> error;
    };

    statistics_measurement(diagram_cfg * cfg, double tau_cutoff, bool split_sign = false, std::size_t n_batches = 256);

    void measure();

    // every quantity of summaryNames() as a function of the batch means x
    Eigen::VectorXd quantities(const Eigen::VectorXd & x) const;

    // Throws simplemc::simplemc_exception with fewer than 2 full batches.
    auto jackknifeStats() const {
        return simplemc::jackknife([this](const Eigen::VectorXd & x){ return quantities(x); }, acc);
    }

    result normalised() const;

    // the reported quantities: n_int n_ext n_total, the same _signed, with split_sign the same _plus and _minus and
    // fraction_plus
    std::vector<std::string> summaryNames() const;
    // This rank's own result (before simplemc_mpi_collect merges the ranks), for the per-chain file: value and
    // jackknife error of each quantity in summaryNames(), in that order; NaNs with fewer than 2 full batches.
    std::vector<double> summary() const;

    // '#' metadata lines, then one row per quantity (n_int, n_ext, n_total): plain err signed err (plus err minus err)
    void write(const std::string & path) const;

    // several ranks (collective): merges the batches of every rank into acc and sums the counters, on every rank
    friend void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, statistics_measurement & m);

    // checkpoint hooks (simplemc's persistent-state channel): the accumulator and the counters; the construction
    // parameters come from the input (main checks that the input still matches the checkpoint)
    template <simplemc::serializer S>
    friend void simplemc_save(S s, const statistics_measurement & m){
        s.save_at("acc", m.acc);
        s.save_at("n_measured", m.n_measured);
        s.save_at("n_negative", m.n_negative);
    }
    template <simplemc::serializer S>
    friend void simplemc_load(const S & s, statistics_measurement & m){
        s.load_at("acc", m.acc);
        s.load_at("n_measured", m.n_measured);
        s.load_at("n_negative", m.n_negative);
    }
};

#endif // !STATISTICS_HPP
