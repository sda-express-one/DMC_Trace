#ifndef THERMALIZATION_CONTROL_HPP
#define THERMALIZATION_CONTROL_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <mpi.h>
#include "measurements/convergence.hpp"

// Ends the thermalization phase of a run when the chains - one per MPI rank - have converged, judged by the improved
// R-hat of Vehtari et al. (measurement::diagnose) on the batch means a convergence_monitor collects on every rank.
//
// Driven from simplemc's run loop: checkpoint() from run_callbacks::on_checkpoint, thermalized() from stop_when.
// Checkpoints must be step-based (simulation_params::checkpoint_after_steps, never checkpoint_after_time), so every
// rank reaches checkpoint c after the same number of steps and the collective calls line up; for the same reason the
// thermalization run must be capped by max_steps, not by max_time (a time cap could stop ranks at different
// checkpoints, leaving collectives unmatched). At checkpoint c:
//   1. every rank completes the broadcast of the decision taken at checkpoint c - 1 (posted then by the root, so in
//      practice already arrived) and, if it says "thermalized", stops: thermalized() turns true on every rank at
//      the same checkpoint, no new collective is posted, and run() returns on all ranks at the same step count;
//   2. otherwise every rank posts a non-blocking gather of its window to the root, and a non-blocking broadcast of
//      the decision for checkpoint c. The root waits for all windows (only it may idle, for the slowest rank),
//      evaluates them and fills the broadcast; the other ranks return at once and keep sampling, reading the
//      decision at checkpoint c + 1. The decision thus acts one checkpoint interval late - extra thermalization.
// Thermalization is reached once every monitored scalar passes at `consecutive` successive evaluations (a failure
// resets the count): max(rank-normalised split-R-hat, folded split-R-hat) < rhat_max, and the (total, over all chains)
// bulk-ESS >= max(min_bulk_ess, min_bulk_ess_per_split_chain x split chains), likewise the tail-ESS. The totals are the
// paper's recommendation (rank-normalised ESS > 400); its "50 per split chain" is the same 400 for its minimum of 4
// chains; the per-split-chain floor (default 50, the paper's figure) decides above 4 ranks. With the monitor's default
// 1024 batches (256-512 draws per split chain) the floor asks for an ESS/draws ratio of 0.1-0.2. Windows with fewer than 8 batches (the start of the run) never pass. The window holds between
// n_batches/2 and n_batches batch means per chain, i.e. at least n_batches/4 draws per split chain, and an ESS cannot
// exceed the number of draws: the constructor refuses settings with n_batches/4 below the per-split-chain floors.
// The totals need enough chains (2 x ranks x n_batches/4 draws >= min_*_ess): the chain count is known only from the
// windows, so this is not checked here (diagramc warns).
//
// Wall-time budget: with setDeadline(t) (an absolute MPI_Wtime value), the root also ends the phase - without
// thermalization - at the first checkpoint it evaluates after t; the decision travels like a "thermalized" one, so
// every rank stops at the same checkpoint (a per-rank time limit would not). thermalized() and timedOut() tell the
// two apart, stopped() is either: use it in stop_when.
//
// Usage (the diagram state carries over from thermalization to data collection):
//     convergence_monitor mon {&cfg, {}};
//     auto therm {simplemc::measurement_set{simplemc::measurement{mon, "convergence"}}};
//     thermalization_control ctl {MPI_COMM_WORLD, mon.names(), mon.maxWindowBatches()};
//     simplemc::simulation_params tp {.max_steps = cap, .max_time = huge, .checkpoint_after_steps = interval};
//     auto & m {therm.get<0>().value()};                    // the monitor inside the set (index 0)
//     simplemc::run(rng, kernel, therm, tp, simplemc::run_callbacks{
//         .on_checkpoint = [&](const auto & ctx){ ctl.checkpoint(m, ctx.steps_done); },
//         .stop_when     = [&](const auto &){ return ctl.stopped(); }});
//     if (!ctl.thermalized()) { /* cap reached: warn or abort */ }
//     simplemc::run(rng, kernel, production, pp, ...);
struct convergence_monitor;

class thermalization_control {
public:
    struct Options {
        double rhat_max {1.01};
        double min_bulk_ess {400.};                    // total over all chains: the paper's recommendation
        double min_tail_ess {400.};
        double min_bulk_ess_per_split_chain {50.};     // floor scaling with the number of chains (binds above 4 ranks)
        double min_tail_ess_per_split_chain {50.};
        int consecutive {5};
        int root {0};
    };

    // one evaluation (root only), kept for logging
    struct Record {
        int checkpoint {0};
        std::uint64_t steps {0};
        std::size_t batches {0};                                // per rank, in the window
        std::vector<measurement::ConvergenceResult> results;    // one per scalar
        bool pass {false};
        int consecutive {0};
    };

    thermalization_control(MPI_Comm comm, std::vector<std::string> names, std::size_t max_window_batches, Options opt);
    thermalization_control(MPI_Comm comm, std::vector<std::string> names, std::size_t max_window_batches)
        : thermalization_control(comm, std::move(names), max_window_batches, Options{}) {}
    ~thermalization_control();

    thermalization_control(const thermalization_control &) = delete;
    thermalization_control & operator=(const thermalization_control &) = delete;

    // from run_callbacks::on_checkpoint, collectively on every rank
    void checkpoint(const convergence_monitor & monitor, std::uint64_t steps);
    // the same with an explicit window (scalar-major, `batches` per scalar), e.g. for tests
    void checkpoint(const std::vector<double> & window, std::size_t batches, std::uint64_t steps);

    // optional wall-time budget: an absolute MPI_Wtime value (the clock of the root decides); none by default
    void setDeadline(double wtime) { deadline_ = wtime; }

    // true on every rank from the same checkpoint on: the phase ended by convergence, by the deadline, or either
    bool thermalized() const { return stop_reason_ == 1; }
    bool timedOut() const { return stop_reason_ == 2; }
    bool stopped() const { return stop_; }

    // The pure decision logic (no MPI): windows[r] of rank/chain r, scalar-major with `batches` per scalar. Updates
    // the consecutive count and returns the record; thermalization is reached when record.consecutive >= consecutive.
    Record evaluate(const std::vector<std::vector<double>> & windows, std::size_t batches, std::uint64_t steps);

    const std::vector<Record> & history() const { return history_; }      // root only
    const std::vector<std::string> & names() const { return names_; }
    bool isRoot() const { return rank_ == opt_.root; }

    // root only: one line per evaluation and scalar (checkpoint steps batches scalar R-hat bulk fold ESS_bulk ESS_tail pass consecutive)
    void write(const std::string & path) const;

private:
    MPI_Comm comm_;
    std::vector<std::string> names_;
    std::size_t max_batches_;
    Options opt_;
    int rank_ {0}, size_ {1};

    bool stop_ {false};
    int stop_reason_ {0};                   // 1 thermalized, 2 deadline
    double deadline_ {std::numeric_limits<double>::infinity()};
    int consecutive_ {0};
    int checkpoints_ {0};
    std::vector<Record> history_;

    // non-blocking exchange: send buffer [batches, window...] of fixed length, the root's receive buffer, the
    // decision {stop: 0 continue / 1 thermalized / 2 deadline, consecutive}
    std::vector<double> send_, recv_;
    std::array<int, 2> decision_ {0, 0};
    MPI_Request gather_req_ {MPI_REQUEST_NULL}, bcast_req_ {MPI_REQUEST_NULL};
};

#endif // !THERMALIZATION_CONTROL_HPP
