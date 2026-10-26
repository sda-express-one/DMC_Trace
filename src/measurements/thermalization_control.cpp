#include "measurements/thermalization_control.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include "measurements/convergence_monitor.hpp"

thermalization_control::thermalization_control(MPI_Comm comm, std::vector<std::string> names, std::size_t max_window_batches,
                                               Options opt)
    : comm_(comm), names_(std::move(names)), max_batches_(max_window_batches), opt_(opt)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    assert(opt_.root >= 0 && opt_.root < size_);
    assert(opt_.consecutive >= 1);
    // the window has at least max_window_batches/4 draws per split chain right after a merge of the batches
    const double min_draws {static_cast<double>(max_batches_) / 4.};
    if (min_draws < opt_.min_bulk_ess_per_split_chain || min_draws < opt_.min_tail_ess_per_split_chain) {
        throw std::invalid_argument("thermalization_control: n_batches / 4 is below the required ESS per split chain - "
                                    "the convergence test could never pass (use n_batches >= 4 x the ESS threshold)");
    }
    send_.assign(1 + names_.size() * max_batches_, 0.);
    if (isRoot()) { recv_.assign(send_.size() * static_cast<std::size_t>(size_), 0.); }
}

thermalization_control::~thermalization_control(){
    // normally nothing is pending (no collective is posted once thermalized); after a run stopped by the step or
    // time cap the last gather/broadcast are still open: complete them (every rank destroys its controller)
    if (gather_req_ != MPI_REQUEST_NULL) { MPI_Wait(&gather_req_, MPI_STATUS_IGNORE); }
    if (bcast_req_ != MPI_REQUEST_NULL) { MPI_Wait(&bcast_req_, MPI_STATUS_IGNORE); }
}

void thermalization_control::checkpoint(const convergence_monitor & monitor, std::uint64_t steps){
    std::size_t batches {0};
    const std::vector<double> w {monitor.window(batches)};
    checkpoint(w, batches, steps);
}

void thermalization_control::checkpoint(const std::vector<double> & window, std::size_t batches, std::uint64_t steps){
    if (stop_) { return; }
    if (batches > max_batches_ || window.size() != names_.size() * batches) {
        throw std::runtime_error("thermalization_control::checkpoint: window larger than declared or inconsistent");
    }

    // 1. the decision of the previous checkpoint (the root filled it right after that gather)
    if (bcast_req_ != MPI_REQUEST_NULL) {
        MPI_Wait(&bcast_req_, MPI_STATUS_IGNORE);
        if (decision_[0] != 0) {      // every rank sees the same decision: nothing new posted
            stop_ = true;
            stop_reason_ = decision_[0];
            return;
        }
    }
    if (gather_req_ != MPI_REQUEST_NULL) { MPI_Wait(&gather_req_, MPI_STATUS_IGNORE); }   // non-root: long complete

    // 2. this checkpoint's window to the root, fixed length: [batches, window, padding]
    std::fill(send_.begin(), send_.end(), 0.);
    send_[0] = static_cast<double>(batches);
    std::copy(window.begin(), window.end(), send_.begin() + 1);
    const int count {static_cast<int>(send_.size())};
    MPI_Igather(send_.data(), count, MPI_DOUBLE, isRoot() ? recv_.data() : nullptr, count, MPI_DOUBLE, opt_.root, comm_, &gather_req_);
    ++checkpoints_;

    if (isRoot()) {
        MPI_Wait(&gather_req_, MPI_STATUS_IGNORE);             // all ranks have reached this checkpoint
        std::vector<std::vector<double>> windows(static_cast<std::size_t>(size_));
        std::size_t b {0};
        bool equal {true};
        for (int r {0}; r < size_; ++r) {
            const double * buf {recv_.data() + static_cast<std::size_t>(r) * send_.size()};
            const auto br {static_cast<std::size_t>(buf[0])};
            if (r == 0) { b = br; }
            equal = equal && br == b;
            windows[static_cast<std::size_t>(r)].assign(buf + 1, buf + 1 + names_.size() * br);
        }
        if (!equal) {      // only if ranks run different step counts per checkpoint: never pass on such data
            for (auto & w : windows) { w.clear(); }
            b = 0;
        }
        const Record rec {evaluate(windows, b, steps)};
        const int stop {rec.consecutive >= opt_.consecutive ? 1 : (MPI_Wtime() >= deadline_ ? 2 : 0)};
        decision_ = {stop, rec.consecutive};
    }
    MPI_Ibcast(decision_.data(), 2, MPI_INT, opt_.root, comm_, &bcast_req_);
}

thermalization_control::Record thermalization_control::evaluate(const std::vector<std::vector<double>> & windows, std::size_t batches,
                                                                std::uint64_t steps){
    Record rec;
    rec.checkpoint = static_cast<int>(history_.size()) + 1;
    rec.steps = steps;
    rec.batches = batches;
    bool pass {batches >= 8 && !windows.empty()};
    for (std::size_t k {0}; k < names_.size(); ++k) {
        std::vector<std::vector<double>> chains;
        if (batches >= 8) {
            for (const auto & w : windows) {
                assert(w.size() == names_.size() * batches);
                chains.emplace_back(w.begin() + static_cast<std::ptrdiff_t>(k * batches),
                                    w.begin() + static_cast<std::ptrdiff_t>((k + 1) * batches));
            }
        }
        const measurement::ConvergenceResult r {measurement::diagnose(chains)};
        rec.results.push_back(r);
        // NaN comparisons are false: a scalar without a valid diagnostic never passes
        const double split {static_cast<double>(r.chains)};
        const bool ok {r.rhat < opt_.rhat_max && r.ess_bulk >= std::max(opt_.min_bulk_ess, opt_.min_bulk_ess_per_split_chain * split)
                       && r.ess_tail >= std::max(opt_.min_tail_ess, opt_.min_tail_ess_per_split_chain * split)};
        pass = pass && ok;
    }
    consecutive_ = pass ? consecutive_ + 1 : 0;
    rec.pass = pass;
    rec.consecutive = consecutive_;
    history_.push_back(rec);
    return rec;
}

void thermalization_control::write(const std::string & path) const {
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("thermalization_control::write: cannot open " + path); }
    std::fprintf(f, "# thermalization: R-hat convergence test (Vehtari et al. 2021), %d chains\n", size_);
    std::fprintf(f, "# rhat_max = %.4f   min bulk/tail-ESS = max(%.1f, %.1f x split chains) / max(%.1f, %.1f x split chains)   consecutive = %d\n",
                 opt_.rhat_max, opt_.min_bulk_ess, opt_.min_bulk_ess_per_split_chain, opt_.min_tail_ess, opt_.min_tail_ess_per_split_chain,
                 opt_.consecutive);
    std::fprintf(f, "# thermalized: %s\n", thermalized() ? "yes"
                 : timedOut() ? "no - stopped by the wall-time budget" : "no (not yet, or the step cap ended the run)");
    std::fprintf(f, "# columns: checkpoint steps batches scalar rhat rhat_bulk rhat_fold ess_bulk ess_tail pass consecutive\n");
    for (const Record & r : history_) {
        for (std::size_t k {0}; k < names_.size(); ++k) {
            const measurement::ConvergenceResult & c {r.results[k]};
            std::fprintf(f, "%d %llu %zu %s %.6f %.6f %.6f %.1f %.1f %d %d\n", r.checkpoint, static_cast<unsigned long long>(r.steps),
                         r.batches, names_[k].c_str(), c.rhat, c.rhat_bulk, c.rhat_fold, c.ess_bulk, c.ess_tail, r.pass ? 1 : 0, r.consecutive);
        }
    }
    std::fclose(f);
}
