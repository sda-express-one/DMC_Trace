#include "updates/scl_diagram.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include "comp_method/segment_action.hpp"
#include "comp_method/weight_computation.hpp"
#include "diagram/vertex.hpp"
#include "utils/prop_distribs.hpp"

double scl_diagram_update::attempt(){
    proposed_tau_values.clear();

    const double tau_D {cfg->current_tau_length};
    const int n_vertices {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};

    // a * tau_D = sum_i (E_min,i + Omega_i - mu) D_i. Omega: summed energy of the phonon lines covering the
    // segment - every external line covers the first segment once; a creation vertex opens a line (+omega),
    // an annihilation vertex closes one (-omega), so a wrapping external line counts twice where it covers
    // a segment twice.
    double phonon_rate {0.};
    for (int i {1}; i < cfg->external_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg->external_ph_manager->ptr_vertex_pool[i].linked_vertex};
        assert(v->type == -2);
        phonon_rate += v->ph_energy;
    }
    double a_tau {0.};
    for (const Vertex * ptr {cfg->diagram_head}; ptr != cfg->diagram_tail; ptr = ptr->next) {
        if (ptr != cfg->diagram_head) {
            assert(ptr->type != 0);
            phonon_rate += ptr->type > 0 ? ptr->phononEnergy() : -ptr->phononEnergy();
        }
        const std::array<double, 3> energies {ptr->electronEnergy()};
        const double lowest_energy {*std::min_element(energies.begin(), energies.end())};
        a_tau += (lowest_energy + phonon_rate - cfg->chem_pot) * (ptr->tau_next - ptr->tau);
    }
    const double a {a_tau / tau_D};
    assert(a > 0.);

    // tau_D' ~ Gamma(n + 1, a): the scalar part tau_D^n e^{-a tau_D} of the weight along the scaling ray
    const double tau_D_new {proposal::drawGamma(*rng, n_vertices + 1., a)};
    const double lambda {tau_D_new / tau_D};

    // every time scaled by lambda; rejected beyond tau_max, and if rounding (or a degenerate draw) leaves
    // a segment of zero length or a non-finite time - the negated comparisons also reject a NaN
    proposed_tau_values.push_back(0.);
    Eigen::Matrix3d product_proposed {Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d action {Eigen::Matrix3d::Identity()};
    double shift {0.};
    for (const Vertex * ptr {cfg->diagram_head}; ptr != cfg->diagram_tail; ptr = ptr->next) {
        const double tau_end {lambda * ptr->tau_next};
        if (!(tau_end > proposed_tau_values.back()) || !(tau_end <= cfg->tau_max)) {
            return -1.;
        }
        weight::setSegmentAction(action, shift, ptr->electronEnergy(), tau_end - proposed_tau_values.back());
        product_proposed = product_proposed * ptr->vertex_wf_component * action.diagonal().asDiagonal();
        proposed_tau_values.push_back(tau_end);
    }

    // Acceptance ratio. In (tau_D, u) the target along the ray is tau_D^n T~(tau_D) e^{-a tau_D} (the
    // shifts S = sum E_min,i D_i, mu tau_D and the phonon propagators make up e^{-a tau_D}; couplings and
    // overlaps are unchanged), and the proposal Gamma(n + 1, a) is tau_D'^n e^{-a tau_D'} up to
    // normalisation, the same for the reverse move. Everything but the band-normalised traces cancels.
    return sign.take(product_proposed.trace() / cfg->diagram_head->right_component.trace());
}

void scl_diagram_update::accept(){
    // every segment takes its new times and (band-normalised) action (head at 0, tail at tau_D')
    std::size_t j {0};
    for (Vertex * ptr {cfg->diagram_head}; ptr != cfg->diagram_tail; ptr = ptr->next, ++j) {
        ptr->tau = proposed_tau_values[j];
        ptr->tau_next = proposed_tau_values[j + 1];
        ptr->computeElPropAction();
    }
    assert(j + 1 == proposed_tau_values.size());

    cfg->diagram_tail->tau = proposed_tau_values.back();
    cfg->current_tau_length = proposed_tau_values.back();

    // every line's length scaled: refresh its propagator (set on both ends); the even slot of each pair is
    // the line's creation vertex
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
