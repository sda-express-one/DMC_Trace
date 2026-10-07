#include "updates/str_diagram.hpp"
#include "comp_method/weight_computation.hpp"
#include "diagram/vertex.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace {
    // diag(e^{-(E_n - E_min) d}): a segment's action with its lowest-band exponential divided out
    Eigen::DiagonalMatrix<double, 3> bandNormalised(const std::array<double, 3> & energies, double e_min, double duration){
        return Eigen::DiagonalMatrix<double, 3>(std::exp(-(energies[0] - e_min)*duration),
                                                std::exp(-(energies[1] - e_min)*duration),
                                                std::exp(-(energies[2] - e_min)*duration));
    }
}

double str_diagram_update::attempt(){
    proposed_weights.clear();
    proposed_tau_values.clear();

    // Omega: summed energy of the phonon lines covering the current segment. Every external line covers
    // the first segment once (in both orientations); crossing a creation vertex opens a line (+omega),
    // an annihilation vertex closes one (-omega), so a wrapping external line counts twice where it
    // covers a segment twice.
    double phonon_rate {0.};
    for(int i {1}; i < this->cfg->external_ph_manager->current_length; i += 2){
        const Vertex * v {this->cfg->external_ph_manager->ptr_vertex_pool[i].linked_vertex};
        assert(v->type == -2);
        phonon_rate += v->ph_energy;
    }

    // band-normalised products of the proposed and of the current diagram, built in the same walk
    Eigen::Matrix3d product_proposed {Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d product_current {Eigen::Matrix3d::Identity()};

    double tau_previous {0.};
    proposed_tau_values.push_back(0.);

    for (Vertex * ptr {cfg->diagram_head}; ptr != cfg->diagram_tail; ptr = ptr->next) {
        if (ptr != cfg->diagram_head) {
            assert(ptr->type != 0);
            phonon_rate += ptr->type > 0 ? ptr->phononEnergy() : -ptr->phononEnergy();
        }

        const std::array<double, 3> energies {ptr->electronEnergy()};
        const double lowest_energy {*std::min_element(energies.begin(), energies.end())};

        // the duration's proposal rate depends only on the topology and the momenta, not on any
        // duration, so the reverse move draws from the same density
        const double rate {lowest_energy - cfg->chem_pot + phonon_rate};
        assert(rate > 0.);

        const double tau {tau_previous - std::log(1 - std_unif(*rng))/rate};

        // u = 0 gives a zero-length segment (probability zero, but possible in floating point), and the
        // diagram may not grow beyond tau_max; the negated comparison also rejects a NaN tau
        if (!(tau > tau_previous) || tau > cfg->tau_max) {
            return -1.;
        }

        const double duration_new {tau - tau_previous};
        const double duration_current {ptr->tau_next - ptr->tau};

        // only the action changes: momenta, bases, masses and overlaps are those of the current diagram
        weight::ProposedVertexWeight current_weight;
        current_weight.el_prop_action(0,0) = std::exp(-energies[0]*duration_new);
        current_weight.el_prop_action(1,1) = std::exp(-energies[1]*duration_new);
        current_weight.el_prop_action(2,2) = std::exp(-energies[2]*duration_new);
        proposed_weights.push_back(current_weight);
        proposed_tau_values.push_back(tau);

        product_proposed = product_proposed * ptr->vertex_wf_component * bandNormalised(energies, lowest_energy, duration_new);
        product_current = product_current * ptr->vertex_wf_component * bandNormalised(energies, lowest_energy, duration_current);

        tau_previous = tau;
    }

    // Acceptance ratio. In the durations D_i (Jacobian 1 from the vertex times) the weight goes as
    // T(D) prod_i e^{-(Omega_i - mu) D_i} (phonon propagators e^{-omega l} with l the sum of the covered
    // durations, and e^{mu tau_D}), and the proposal is prod_i a_i e^{-a_i D_i} with
    // a_i = E_min,i - mu + Omega_i, the same for the reverse move. Their ratio leaves
    // T(D')/T(D) prod_i e^{E_min,i (D'_i - D_i)}: the ratio of the two traces built with band-normalised
    // actions, where no absolute time or large exponent appears.
    return sign.take(product_proposed.trace() / product_current.trace());
}

void str_diagram_update::accept(){
    // every segment takes its new times and action (head at 0, tail at the new tau_D)
    std::size_t j {0};
    for (Vertex * ptr {cfg->diagram_head}; ptr != cfg->diagram_tail; ptr = ptr->next, ++j) {
        ptr->tau = proposed_tau_values[j];
        ptr->tau_next = proposed_tau_values[j + 1];
        ptr->el_prop_action = proposed_weights[j].el_prop_action;
    }
    assert(j + 1 == proposed_tau_values.size());

    cfg->diagram_tail->tau = proposed_tau_values.back();
    cfg->current_tau_length = proposed_tau_values.back();

    // every line's length changed: refresh its propagator (set on both ends); the even slot of each
    // pair is the line's creation vertex
    for (int i {0}; i < cfg->internal_ph_manager->current_length; i += 2) {
        cfg->internal_ph_manager->selectVertex(i)->computeInternalPhPropAction();
    }
    for (int i {0}; i < cfg->external_ph_manager->current_length; i += 2) {
        cfg->external_ph_manager->selectVertex(i)->computeExternalPhPropAction(cfg->current_tau_length);
    }

    weight::LKMatrix::computeRightSide(cfg->diagram_head, cfg->diagram_tail);
    weight::LKMatrix::computeLeftSide(cfg->diagram_tail, cfg->diagram_head);

    sign.accepted(cfg);
}
