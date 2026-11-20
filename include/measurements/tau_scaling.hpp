#ifndef TAU_SCALING_HPP
#define TAU_SCALING_HPP

#include <array>
#include <Eigen/Core>
#include "diagram/diagram_config.hpp"
#include "diagram/vertex.hpp"

// The pieces of the tau-scaling energy estimator (gs_energy_measurement, gs_energy_channels_measurement). Scaling every
// time of a diagram of length tau by lambda gives, per diagram,
//     tau X = -n + sum_lines omega l + sum_segments dtau_i tr(S_i) / tr P,
//     S_i = L_i wf_i E_i A_i wf_{i+1} R_{i+1}
// with E_i the segment's band energies inserted next to its action (see gs_energy.hpp). S_i is the full product
// P = diagram_head->right_component with that insertion, in the same (head's band) basis, so any linear functional of
// P - the trace, or a projected trace tr(Q P) - has its tau derivative from the same S_i.
namespace tau_scaling {

// n: one integrated time per vertex
inline int nVertices(const diagram_cfg & cfg){
    return cfg.internal_ph_manager->current_length + cfg.external_ph_manager->current_length;
}

// sum_lines omega l; the even slot of each pair is the line's creation vertex (layout rule). l = tau2 - tau1 for an
// internal line and tau - tau_cre + tau_ann for an external (wrapped) one.
inline double phononTerm(const diagram_cfg & cfg){
    const double tau {cfg.current_tau_length};
    double phonon {0.};
    for (int i {0}; i < cfg.internal_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg.internal_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (v->conj_vertex->tau - v->tau);
    }
    for (int i {0}; i < cfg.external_ph_manager->current_length; i += 2) {
        const Vertex * v {cfg.external_ph_manager->selectVertex(i)};
        phonon += v->ph_energy * (tau - v->tau + v->conj_vertex->tau);
    }
    return phonon;
}

// Calls f(dtau_i, S_i) for every segment, head to tail. S_i is passed as the Eigen product expression (not evaluated),
// so that a caller taking only its trace performs exactly the arithmetic of the original gs_energy estimator.
template <typename F>
inline void forEachSegment(const diagram_cfg & cfg, F && f){
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) {
        const std::array<double, 3> e {v->electronEnergy()};
        const Eigen::Matrix3d e_action {Eigen::Vector3d(e[0]*v->el_prop_action(0,0), e[1]*v->el_prop_action(1,1),
                                                        e[2]*v->el_prop_action(2,2)).asDiagonal()};
        f(v->tau_next - v->tau,
          v->left_component * v->vertex_wf_component * e_action * v->next->vertex_wf_component * v->next->right_component);
    }
}

} // namespace tau_scaling

#endif // !TAU_SCALING_HPP
