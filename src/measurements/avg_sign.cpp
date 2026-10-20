#include "measurements/avg_sign.hpp"
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

avg_sign_measurement::avg_sign_measurement(diagram_cfg * cfg, double tau_min, bool tau_fixed, std::size_t n_batches)
    : cfg(cfg),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      acc(1, n_batches)
{
    assert(cfg != nullptr);
}

void avg_sign_measurement::measure(){
    if (!tau_fixed && cfg->current_tau_length < tau_min) {
        return;
    }
    const double trace {cfg->diagram_head->right_component.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    const double s {trace < 0. ? -1. : 1.};
    acc.accumulate(Eigen::Matrix<double, 1, 1>(s));
    ++n_measured;
    if (s < 0.) { ++n_negative; }
}

avg_sign_measurement::result avg_sign_measurement::normalised() const {
    const auto jk {jackknifeMean()};
    return result{static_cast<long double>(jk.naive_mean()(0)), static_cast<long double>(jk.stderror()(0))};
}

void avg_sign_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("avg_sign_measurement::write: cannot open " + path); }

    std::fprintf(f, "# average sign <s> = sum W / sum |W|\n");
    std::fprintf(f, "# %s", tau_fixed ? "tau_fixed\n" : "");
    if (!tau_fixed) { std::fprintf(f, "tau_min = %.10f\n", tau_min); }
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_negative = %llu (fraction %.6g)   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_negative),
                 n_measured > 0 ? static_cast<double>(n_negative) / static_cast<double>(n_measured) : 0.,
                 acc.batches().size(), static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# columns: sign err\n");
    std::fprintf(f, "%.15Le %.6Le\n", r.sign, r.sign_error);
    std::fclose(f);
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, avg_sign_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    simplemc::mpi::all_reduce_in_place(m.n_measured, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
}

std::vector<std::string> avg_sign_measurement::summaryNames(){
    return {"sign"};
}

std::vector<double> avg_sign_measurement::summary() const {
    try {
        const result r {normalised()};
        return {static_cast<double>(r.sign), static_cast<double>(r.sign_error)};
    }
    catch (const std::exception &) {      // fewer than 2 full batches on this rank
        return std::vector<double>(2, std::numeric_limits<double>::quiet_NaN());
    }
}
