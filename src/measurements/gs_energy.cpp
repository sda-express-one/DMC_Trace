#include "measurements/gs_energy.hpp"
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

gs_energy_measurement::gs_energy_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed,
                                             std::size_t n_batches, bool sign_free)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      sign_free(sign_free),
      acc(sign_free ? 3 : 2, n_batches)
{
    assert(cfg != nullptr);
}

double gs_energy_measurement::estimate() const {
    const double tau {cfg->current_tau_length};
    const double trace {cfg->diagram_head->right_component.trace()};
    assert(std::isfinite(trace) && trace != 0.);

    // -n: one integrated time per vertex
    const int n_vertices {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};

    // sum_lines omega l; the even slot of each pair is the line's creation vertex (layout rule)
    double phonon {0.};
    for (int i {0}; i < cfg->internal_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg->internal_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (v->conj_vertex->tau - v->tau);
    }
    for (int i {0}; i < cfg->external_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg->external_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (tau - v->tau + v->conj_vertex->tau);
    }

    // sum_segments dtau_i tr(L_i wf_i E_i A_i wf_{i+1} R_{i+1}): the full trace with the segment's
    // energies inserted next to its action (diagonal, so they commute)
    double electron {0.};
    for (const Vertex * v {cfg->diagram_head}; v != cfg->diagram_tail; v = v->next) {
        const std::array<double, 3> e {v->electronEnergy()};
        const Eigen::Matrix3d e_action {Eigen::Vector3d(e[0]*v->el_prop_action(0,0), e[1]*v->el_prop_action(1,1),
                                                        e[2]*v->el_prop_action(2,2)).asDiagonal()};
        electron += (v->tau_next - v->tau)
                  * (v->left_component * v->vertex_wf_component * e_action * v->next->vertex_wf_component * v->next->right_component).trace();
    }

    return (-static_cast<double>(n_vertices) + phonon + electron / trace) / tau;
}

void gs_energy_measurement::measure(){
    if (!tau_fixed && cfg->current_tau_length < tau_min) {
        return;
    }
    const double trace {cfg->diagram_head->right_component.trace()};
    if (!std::isfinite(trace) || trace == 0.) {     // a zero trace has probability zero
        return;
    }
    const double s {trace < 0. ? -1. : 1.};
    const double X {estimate()};
    if (sign_free) { acc.accumulate(Eigen::Vector3d(s * X, s, X)); }
    else { acc.accumulate(Eigen::Vector2d(s * X, s)); }
    ++n_measured;
    if (s < 0.) { ++n_negative; }
}

gs_energy_measurement::result gs_energy_measurement::normalised() const {
    const auto jk {jackknifeRatio()};
    const Eigen::VectorXd mean {jk.naive_mean()}, err {jk.stderror()};
    result r {static_cast<long double>(mean(0)), static_cast<long double>(err(0)),
              std::numeric_limits<long double>::quiet_NaN(), std::numeric_limits<long double>::quiet_NaN()};
    if (sign_free) {
        r.energy_sf = static_cast<long double>(mean(1));
        r.energy_sf_error = static_cast<long double>(err(1));
    }
    return r;
}

void gs_energy_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("gs_energy_measurement::write: cannot open " + path); }

    std::fprintf(f, "# ground-state energy, tau-scaling estimator\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   %s", p[0], p[1], p[2], tau_fixed ? "tau_fixed\n" : "");
    if (!tau_fixed) { std::fprintf(f, "tau_min = %.10f\n", tau_min); }
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   n_negative = %llu   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 static_cast<unsigned long long>(n_negative),
                 acc.batches().size(), static_cast<unsigned long long>(acc.batch_count()));
    if (sign_free) {
        std::fprintf(f, "# E_sf: sign-free energy <X> of the sampled |W| ensemble (E_sf <= E_0; <s> ~ exp(-(E_0 - E_sf) tau_D))\n");
        std::fprintf(f, "# columns: E err E_sf err\n");
        std::fprintf(f, "%.15Le %.6Le %.15Le %.6Le\n", r.energy, r.energy_error, r.energy_sf, r.energy_sf_error);
    }
    else {
        std::fprintf(f, "# columns: E err\n");
        std::fprintf(f, "%.15Le %.6Le\n", r.energy, r.energy_error);
    }
    std::fclose(f);
}

void simplemc_mpi_collect(const simplemc::mpi::communicator & comm, gs_energy_measurement & m){
    simplemc_mpi_collect(comm, m.acc);
    simplemc::mpi::all_reduce_in_place(m.n_measured, MPI_SUM, comm);
    simplemc::mpi::all_reduce_in_place(m.n_negative, MPI_SUM, comm);
}

std::vector<std::string> gs_energy_measurement::summaryNames() const {
    if (sign_free) { return {"E", "E_sf"}; }
    return {"E"};
}

std::vector<double> gs_energy_measurement::summary() const {
    try {
        const result r {normalised()};
        std::vector<double> out {static_cast<double>(r.energy), static_cast<double>(r.energy_error)};
        if (sign_free) {
            out.push_back(static_cast<double>(r.energy_sf));
            out.push_back(static_cast<double>(r.energy_sf_error));
        }
        return out;
    }
    catch (const std::exception &) {      // fewer than 2 full batches on this rank
        return std::vector<double>(2 * summaryNames().size(), std::numeric_limits<double>::quiet_NaN());
    }
}
