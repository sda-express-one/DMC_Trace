#include "measurements/gs_energy_channels.hpp"
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
#include "diagram/vertex.hpp"
#include "measurements/tau_scaling.hpp"

gs_energy_channels_measurement::gs_energy_channels_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min,
                                                               bool tau_fixed, std::size_t n_batches)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      group_order(static_cast<int>(symmetry::littleGroup(p).size())),
      channels(symmetry::channelsOf(p)),
      acc(2 * channels.size() + 1, n_batches)       // channels is declared (so initialised) before acc
{
    assert(cfg != nullptr);
}

Eigen::VectorXd gs_energy_channels_measurement::estimate() const {
    const double tau {cfg->current_tau_length};
    const Eigen::Matrix3d & P {cfg->diagram_head->right_component};
    const double abs_trace {std::abs(P.trace())};
    assert(std::isfinite(abs_trace) && abs_trace != 0.);
    assert(cfg->external_ph_manager->current_length == 0);

    // the projectors in the head's band basis: orbital M = U P U^T, so tr(Q M) = tr(U^T Q U P)
    const Eigen::Matrix3d & U {cfg->diagram_head->baseWF};
    const int nc {nChannels()};
    std::vector<Eigen::Matrix3d> Qt(static_cast<std::size_t>(nc));
    for (int c {0}; c < nc; ++c) { Qt[static_cast<std::size_t>(c)] = U.transpose() * channels[static_cast<std::size_t>(c)].Q * U; }

    const double scalar {-static_cast<double>(tau_scaling::nVertices(*cfg)) + tau_scaling::phononTerm(*cfg)};
    Eigen::VectorXd electron {Eigen::VectorXd::Zero(nc)};
    tau_scaling::forEachSegment(*cfg, [&](double dtau, const auto & S){
        const Eigen::Matrix3d Sm {S};
        for (int c {0}; c < nc; ++c) { electron(c) += dtau * (Qt[static_cast<std::size_t>(c)] * Sm).trace(); }
    });

    Eigen::VectorXd out(2 * nc);
    for (int c {0}; c < nc; ++c) {
        const double proj {(Qt[static_cast<std::size_t>(c)] * P).trace()};
        out(2*c) = (scalar * proj + electron(c)) / (tau * abs_trace);
        out(2*c + 1) = proj / abs_trace;
    }
    return out;
}

void gs_energy_channels_measurement::measure(){
    if (!tau_fixed && cfg->current_tau_length < tau_min) {
        return;
    }
    const double trace {cfg->diagram_head->right_component.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    const int nc {nChannels()};
    Eigen::VectorXd sample {Eigen::VectorXd::Zero(2 * nc + 1)};
    ++n_measured;
    if (cfg->external_ph_manager->current_length == 0) {
        sample.head(2 * nc) = estimate();
        sample(2 * nc) = 1.;
        ++n_vacuum;
        if (trace < 0.) { ++n_negative; }
    }
    acc.accumulate(sample);
}

Eigen::VectorXd gs_energy_channels_measurement::ratio(const Eigen::VectorXd & x) const {
    const int nc {nChannels()};
    Eigen::VectorXd out(2 * nc + 1);
    for (int c {0}; c < nc; ++c) {
        out(c) = x(2*c) / x(2*c + 1);
        out(nc + c) = x(2*c + 1);
    }
    out(2 * nc) = x(2 * nc);
    return out;
}

gs_energy_channels_measurement::result gs_energy_channels_measurement::normalised() const {
    const auto jk {jackknifeRatio()};
    const Eigen::VectorXd mean {jk.naive_mean()}, err {jk.stderror()};
    const int nc {nChannels()};
    result r;
    for (int c {0}; c < nc; ++c) {
        r.energy.push_back(static_cast<long double>(mean(c)));
        r.energy_error.push_back(static_cast<long double>(err(c)));
        r.share.push_back(static_cast<long double>(mean(nc + c)));
        r.share_error.push_back(static_cast<long double>(err(nc + c)));
    }
    r.vacuum = static_cast<long double>(mean(2 * nc));
    r.vacuum_error = static_cast<long double>(err(2 * nc));
    return r;
}

void gs_energy_channels_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("gs_energy_channels_measurement::write: cannot open " + path); }

    const double p2 {p[0]*p[0] + p[1]*p[1] + p[2]*p[2]};
    std::fprintf(f, "# ground-state energy per symmetry channel of p, tau-scaling estimator of tr(Q_c G(p, tau)), vacuum sector\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   |p|^2 = %.10e   %s", p[0], p[1], p[2], p2, tau_fixed ? "tau_fixed\n" : "");
    if (!tau_fixed) { std::fprintf(f, "tau_min = %.10f\n", tau_min); }
    std::fprintf(f, "# little group order = %d   channels = %d\n", group_order, nChannels());
    for (int c {0}; c < nChannels(); ++c) {
        const symmetry::Channel & ch {channels[static_cast<std::size_t>(c)]};
        std::fprintf(f, "# channel %d: dim %d   bare lambda =", c, ch.dim);
        for (const double b : ch.bare) { std::fprintf(f, " %.6f", b); }
        std::fprintf(f, "   bare energy lambda p^2 =");
        for (const double b : ch.bare) { std::fprintf(f, " %.6e", b * p2); }
        std::fprintf(f, "   orbital basis:");
        for (int k {0}; k < ch.dim; ++k) {
            std::fprintf(f, " (%+.6f, %+.6f, %+.6f)", ch.basis(0, k), ch.basis(1, k), ch.basis(2, k));
        }
        std::fprintf(f, "\n");
    }
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_vacuum = %llu   n_negative (vacuum) = %llu   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_vacuum), static_cast<unsigned long long>(n_negative),
                 acc.batches().size(), static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# vacuum fraction = %.10Le +- %.3Le   (share: <r_c> = <tr(Q_c U P U^T) / |tr P|> per measured sample; "
                    "the shares sum to the vacuum-sector <sign>)\n", r.vacuum, r.vacuum_error);
    std::fprintf(f, "# columns: channel dim E err share err\n");
    for (int c {0}; c < nChannels(); ++c) {
        std::fprintf(f, "%d %d %.15Le %.6Le %.10Le %.6Le\n", c, channels[static_cast<std::size_t>(c)].dim,
                     r.energy[static_cast<std::size_t>(c)], r.energy_error[static_cast<std::size_t>(c)],
                     r.share[static_cast<std::size_t>(c)], r.share_error[static_cast<std::size_t>(c)]);
    }
    std::fclose(f);
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, gs_energy_channels_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    simplemc::mpi::all_reduce_in_place(m.n_measured, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_vacuum, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
}

std::vector<std::string> gs_energy_channels_measurement::summaryNames() const {
    std::vector<std::string> names;
    for (int c {0}; c < nChannels(); ++c) { names.push_back("E_ch" + std::to_string(c)); }
    return names;
}

std::vector<double> gs_energy_channels_measurement::summary() const {
    try {
        const result r {normalised()};
        std::vector<double> out;
        for (int c {0}; c < nChannels(); ++c) {
            out.push_back(static_cast<double>(r.energy[static_cast<std::size_t>(c)]));
            out.push_back(static_cast<double>(r.energy_error[static_cast<std::size_t>(c)]));
        }
        return out;
    }
    catch (const std::exception &) {      // fewer than 2 full batches on this rank
        return std::vector<double>(2 * summaryNames().size(), std::numeric_limits<double>::quiet_NaN());
    }
}
