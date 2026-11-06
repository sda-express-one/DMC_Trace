#include "measurements/statistics.hpp"
#include <mpi.h>
#include <simplemc/mpi/all_reduce.hpp>
#include <simplemc/mpi/communicator.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "diagram/vertex_manager.hpp"

namespace {
    // sample layout: [n_int, n_ext, s, s n_int, s n_ext] (+ [I+, I+ n_int, I+ n_ext] with split_sign)
    constexpr int base_size {5};
    constexpr int split_size {3};
}

statistics_measurement::statistics_measurement(diagram_cfg * cfg, double tau_cutoff, bool split_sign, std::size_t n_batches)
    : cfg(cfg),
      tau_cutoff(tau_cutoff),
      split_sign(split_sign),
      acc(split_sign ? base_size + split_size : base_size, n_batches)
{
    assert(cfg != nullptr);
}

void statistics_measurement::measure(){
    if (cfg->current_tau_length < tau_cutoff) {
        return;
    }
    const double trace {cfg->diagram_head->right_component.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    const double s {trace < 0. ? -1. : 1.};
    // two vertices per line
    const double n_int {static_cast<double>(cfg->internal_ph_manager->current_length) / 2.};
    const double n_ext {static_cast<double>(cfg->external_ph_manager->current_length) / 2.};
    if (split_sign) {
        const double plus {s > 0. ? 1. : 0.};
        Eigen::Matrix<double, base_size + split_size, 1> x;
        x << n_int, n_ext, s, s * n_int, s * n_ext, plus, plus * n_int, plus * n_ext;
        acc.accumulate(x);
    }
    else {
        Eigen::Matrix<double, base_size, 1> x;
        x << n_int, n_ext, s, s * n_int, s * n_ext;
        acc.accumulate(x);
    }
    ++n_measured;
    if (s < 0.) { ++n_negative; }
}

Eigen::VectorXd statistics_measurement::quantities(const Eigen::VectorXd & x) const {
    const double nan {std::numeric_limits<double>::quiet_NaN()};
    // a 0/0 (no sample of that kind) is NaN, not an exception or an infinity
    const auto ratio {[nan](double a, double b){ return b != 0. ? a / b : nan; }};
    Eigen::VectorXd q(split_sign ? 13 : 6);
    q(0) = x(0);
    q(1) = x(1);
    q(2) = x(0) + x(1);
    q(3) = ratio(x(3), x(2));
    q(4) = ratio(x(4), x(2));
    q(5) = ratio(x(3) + x(4), x(2));
    if (split_sign) {
        const double fp {x(5)}, fm {1. - x(5)};
        q(6) = ratio(x(6), fp);
        q(7) = ratio(x(7), fp);
        q(8) = ratio(x(6) + x(7), fp);
        q(9) = ratio(x(0) - x(6), fm);
        q(10) = ratio(x(1) - x(7), fm);
        q(11) = ratio(x(0) + x(1) - x(6) - x(7), fm);
        q(12) = fp;
    }
    return q;
}

statistics_measurement::result statistics_measurement::normalised() const {
    const auto jk {jackknifeStats()};
    const Eigen::VectorXd mean {jk.naive_mean()}, err {jk.stderror()};
    return result{std::vector<double>(mean.data(), mean.data() + mean.size()),
                  std::vector<double>(err.data(), err.data() + err.size())};
}

std::vector<std::string> statistics_measurement::summaryNames() const {
    std::vector<std::string> names {"n_int", "n_ext", "n_total", "n_int_signed", "n_ext_signed", "n_total_signed"};
    if (split_sign) {
        names.insert(names.end(), {"n_int_plus", "n_ext_plus", "n_total_plus", "n_int_minus", "n_ext_minus", "n_total_minus",
                                   "fraction_plus"});
    }
    return names;
}

std::vector<double> statistics_measurement::summary() const {
    try {
        const result r {normalised()};
        std::vector<double> out;
        out.reserve(2 * r.value.size());
        for (std::size_t i {0}; i < r.value.size(); ++i) {
            out.push_back(r.value[i]);
            out.push_back(r.error[i]);
        }
        return out;
    }
    catch (const std::exception &) {      // fewer than 2 full batches on this rank
        return std::vector<double>(2 * summaryNames().size(), std::numeric_limits<double>::quiet_NaN());
    }
}

void statistics_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("statistics_measurement::write: cannot open " + path); }

    std::fprintf(f, "# diagram statistics: average number of internal and external phonon lines\n");
    std::fprintf(f, "# tau_cutoff = %.10f (diagrams with tau_D >= tau_cutoff)\n", tau_cutoff);
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_negative = %llu (fraction %.6g)   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_negative),
                 n_measured > 0 ? static_cast<double>(n_negative) / static_cast<double>(n_measured) : 0.,
                 acc.batches().size(), static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# plain: mean over the sampled (sign-free) ensemble; signed: <s n> / <s>%s\n",
                 split_sign ? "; plus/minus: mean over the positive/negative diagrams" : "");
    if (split_sign) {
        std::fprintf(f, "# fraction_plus = %.15e +- %.6e\n", r.value[12], r.error[12]);
        std::fprintf(f, "# columns: quantity plain err signed err plus err minus err\n");
    }
    else {
        std::fprintf(f, "# columns: quantity plain err signed err\n");
    }
    const char * rows[3] {"n_int", "n_ext", "n_total"};
    for (std::size_t i {0}; i < 3; ++i) {
        std::fprintf(f, "%s %.15e %.6e %.15e %.6e", rows[i], r.value[i], r.error[i], r.value[i + 3], r.error[i + 3]);
        if (split_sign) {
            std::fprintf(f, " %.15e %.6e %.15e %.6e", r.value[i + 6], r.error[i + 6], r.value[i + 9], r.error[i + 9]);
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, statistics_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    simplemc::mpi::all_reduce_in_place(m.n_measured, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
}
