#include "measurements/gs_energy.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "diagram/vertex.hpp"

gs_energy_measurement::gs_energy_measurement(diagram_cfg * cfg, std::array<double, 3> p, double tau_min, bool tau_fixed,
                                             std::size_t n_batches)
    : cfg(cfg),
      p(p),
      tau_min(tau_fixed ? -1. : tau_min),
      tau_fixed(tau_fixed),
      acc(2, n_batches)
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
    const Eigen::Vector2d sample {s * estimate(), s};
    acc.accumulate(sample);
    ++n_measured;
}

gs_energy_measurement::result gs_energy_measurement::normalised() const {
    const auto jk {jackknifeRatio()};
    return result{static_cast<long double>(jk.naive_mean()(0)), static_cast<long double>(jk.stderror()(0))};
}

void gs_energy_measurement::write(const std::string & path) const {
    const result r {normalised()};
    FILE * f {std::fopen(path.c_str(), "w")};
    if (f == nullptr) { throw std::runtime_error("gs_energy_measurement::write: cannot open " + path); }

    std::fprintf(f, "# ground-state energy, tau-scaling estimator\n");
    std::fprintf(f, "# p = %.10f %.10f %.10f   %s", p[0], p[1], p[2], tau_fixed ? "tau_fixed\n" : "");
    if (!tau_fixed) { std::fprintf(f, "tau_min = %.10f\n", tau_min); }
    std::fprintf(f, "# n_samples = %llu   n_measured = %llu   batches = %zu x %llu samples\n",
                 static_cast<unsigned long long>(acc.count()), static_cast<unsigned long long>(n_measured),
                 acc.batches().size(), static_cast<unsigned long long>(acc.batch_count()));
    std::fprintf(f, "# columns: E err\n");
    std::fprintf(f, "%.15Le %.6Le\n", r.energy, r.energy_error);
    std::fclose(f);
}
