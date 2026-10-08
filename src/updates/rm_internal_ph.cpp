
// [N_MODES NORMALIZATION - TO BE VERIFIED]
// Inverse of add_*_ph, whose proposal draws the new line's phonon mode uniformly (probability
// 1/N_modes): the ratio carries a factor 1/N_modes (and add_*_ph's N_modes), so that the add x rm
// product stays 1 and the diagrams sum over the mode of every line.

#include "updates/rm_internal_ph.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <Eigen/Core>
#include "diagram/vertex_manager.hpp"
#include "comp_method/weight_computation.hpp"

double rm_int_ph_update::attempt(){
    proposed_weights.clear();

    if(cfg->internal_ph_manager->current_length < 2){
        return -1.;
    }

    // choose random vertex pointer
    ptr_one = cfg->internal_ph_manager->chooseOutgoingVertex()->linked_vertex;
    assert(ptr_one != nullptr);

    ptr_two = ptr_one->conj_vertex;
    assert(ptr_two != nullptr);
    
    const double tau_one {ptr_one->tau};
    const double tau_two {ptr_two->tau};

    const std::array<double, 3> w_to_reject {ptr_one->w};

    Vertex * ptr {ptr_one};
    std::array<double, 3> p_fin {0., 0., 0.};
    double p_fin_sq {0.};

    // product of the staged segments, accumulated left to right as each one is staged, and the sum of
    // their action shifts (the actions are band-normalised: full action = e^{-shift} * action)
    Eigen::Matrix3d new_matrix_product {Eigen::Matrix3d::Identity()};
    double shift_new {0.};
    auto fold = [&new_matrix_product, &shift_new](const weight::ProposedVertexWeight & w){
        new_matrix_product = new_matrix_product * w.vertex_wf_component * w.el_prop_action.diagonal().asDiagonal();
        shift_new += w.action_shift;
    };

    Eigen::Matrix<double, 4, 3> eigensolution_wrapper;
    std::array<double, 3> eigenvalues {1., 1., 1.};

    Vertex * ptr_prev = ptr_one->prev;

    do {
        weight::ProposedVertexWeight current_new_weight;
        
        if(ptr_one->next == ptr_two){
            current_new_weight.baseWF = ptr_prev->baseWF;
            current_new_weight.vertex_wf_component = Eigen::Matrix3d::Identity();

            p_fin = ptr_prev->k;
            current_new_weight.k = p_fin;
            
            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            current_new_weight.eff_masses = ptr_prev->eff_masses;

            weight::setSegmentAction(current_new_weight.el_prop_action, current_new_weight.action_shift, p_fin_sq, ptr_prev->eff_masses, ptr_two->tau_next - ptr_prev->tau);

            proposed_weights.push_back(current_new_weight);
        }
        else if (ptr == ptr_one){
            current_new_weight.baseWF = ptr_prev->baseWF;
            current_new_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            
            p_fin = ptr_prev->k;
            current_new_weight.k = p_fin;

            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            current_new_weight.eff_masses = ptr_prev->eff_masses;

            weight::setSegmentAction(current_new_weight.el_prop_action, current_new_weight.action_shift, p_fin_sq, ptr_prev->eff_masses, ptr_one->tau_next - ptr_prev->tau);

            proposed_weights.push_back(current_new_weight);
        }
        else if (ptr->next == ptr_two){
            current_new_weight.baseWF = ptr_two->baseWF;
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_new_weight.baseWF);

            p_fin = ptr_two->k;
            current_new_weight.k = p_fin;

            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];
        
            current_new_weight.eff_masses = ptr_two->eff_masses;

            weight::setSegmentAction(current_new_weight.el_prop_action, current_new_weight.action_shift, p_fin_sq, ptr_two->eff_masses, ptr_two->tau_next - ptr_two->prev->tau);

            proposed_weights.push_back(current_new_weight);
        }
        else {
            p_fin = {ptr->k[0] + w_to_reject[0], ptr->k[1] + w_to_reject[1], ptr->k[2] + w_to_reject[2]};
            current_new_weight.k = p_fin;
            p_fin_sq = {p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2]};

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
        
            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);

            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_new_weight.baseWF);

            weight::setSegmentAction(current_new_weight.el_prop_action, current_new_weight.action_shift, p_fin_sq, current_new_weight.eff_masses, ptr->tau_next - ptr->tau);

            proposed_weights.push_back(current_new_weight);
        }
        fold(proposed_weights.back());      // every branch stages exactly one segment

        ptr = ptr->next;
    } while (ptr != ptr_two);

    // the segments the move replaces: ptr_prev's (merged with ptr_one's) through ptr_two's
    double shift_old {0.};
    for (const Vertex * v {ptr_prev}; ; v = v->next) {
        shift_old += v->action_shift;
        if (v == ptr_two) { break; }
    }

    // the traces are built from band-normalised actions: the full ratio carries e^{-(S' - S)}, S' - S the
    // change of the summed shifts over the replaced segments (applied in the numerator below)
    const double weight_proposed {
        (new_matrix_product * ptr_two->next->vertex_wf_component * ptr_two->next->right_component * ptr_prev->left_component * ptr_prev->vertex_wf_component).trace()
    };

    const double weight_current {
        this->cfg->diagram_head->right_component.trace() *
        Coupling::Strength::squaredTimesMomentumSquared(ptr_one->ph_energy, ptr_one->diel_response) *
            std::exp(-ptr_one->ph_energy*(ptr_two->tau - ptr_one->tau))
    };

    // inverse of add_int_ph's ratio, with the momentum per dr dOmega in the same way: |g|^2|w|^2 for the
    // line and the sphericalDensity of the proposal that would have drawn w_to_reject
    const double p_A {static_cast<double>(this->cfg->internal_ph_manager->current_length)/2.};
    const double p_B {static_cast<double>(this->cfg->internal_ph_manager->current_length + this->cfg->external_ph_manager->current_length - 1)};

    // [N_MODES NORMALIZATION - TO BE VERIFIED]
    // uniform mode draw in add_*_ph: N_modes in its ratio, 1/N_modes here in rm
    const double n_modes {static_cast<double>(cfg->phonon_mode_manager->num_phonon_modes)};

    const double numerator {
        std::exp(-(shift_new - shift_old)) *
        p_A *
        weight_proposed *
        ptr_one->ph_energy * std::exp(-ptr_one->ph_energy*(tau_two - tau_one)) *
        std::pow(2.*std::numbers::pi, 3) *
        this->w_proposal.sphericalDensity(w_to_reject, tau_two - tau_one)
    };

    const double tau_init_v1 {ptr_one->prev->tau};
    const double tau_end_v1 {ptr_one->next == ptr_two ? ptr_two->tau_next : ptr_one->tau_next};

    const double denominator {
        p_B *
        weight_current *
        (tau_end_v1 -tau_init_v1) *
        Coupling::Parameters::V_unit_cell *
        n_modes   // [N_MODES NORMALIZATION - TO BE VERIFIED]
    };

    return sign.take(numerator/denominator);
}

void rm_int_ph_update::accept(){
    Vertex * ptr_prev {ptr_one->prev};
    const bool adjacent {ptr_one->next == ptr_two};

    // ptr_prev absorbs ptr_one (and, if adjacent, ptr_two too). Its own vertex_wf_component is
    // NOT touched: proposed_weights.front()'s wf was only ever an Identity placeholder used to
    // seed the fold in attempt(), never ptr_prev's real overlap.
    ptr_prev->k = proposed_weights.front().k;
    ptr_prev->eff_masses = proposed_weights.front().eff_masses;
    ptr_prev->baseWF = proposed_weights.front().baseWF;
    ptr_prev->el_prop_action = proposed_weights.front().el_prop_action;
    ptr_prev->action_shift = proposed_weights.front().action_shift;
    // both boundary tau_next values are read from ptr_one/ptr_two before either is removed,
    // so nothing here depends on a ->next chain that's still mid-splice.
    ptr_prev->tau_next = adjacent ? ptr_two->tau_next : ptr_one->tau_next;

    Vertex * last_survivor {ptr_prev};

    if (!adjacent) {
        // surviving interior vertices (proposed_weights[1..size()-2]) don't move in tau, only
        // their momentum-derived state changes; the last one absorbs ptr_two.
        Vertex * ptr {ptr_one->next};
        std::size_t idx {1};
        while (ptr->next != ptr_two) {
            ptr->k = proposed_weights[idx].k;
            ptr->eff_masses = proposed_weights[idx].eff_masses;
            ptr->baseWF = proposed_weights[idx].baseWF;
            ptr->vertex_wf_component = proposed_weights[idx].vertex_wf_component;
            ptr->el_prop_action = proposed_weights[idx].el_prop_action;
            ptr->action_shift = proposed_weights[idx].action_shift;
            ptr = ptr->next;
            ++idx;
        }
        ptr->k = proposed_weights.back().k;
        ptr->eff_masses = proposed_weights.back().eff_masses;
        ptr->baseWF = proposed_weights.back().baseWF;
        ptr->vertex_wf_component = proposed_weights.back().vertex_wf_component;
        ptr->el_prop_action = proposed_weights.back().el_prop_action;
        ptr->action_shift = proposed_weights.back().action_shift;
        ptr->tau_next = ptr_two->tau_next;

        last_survivor = ptr;
    }

    // unregister the line: findPointer() gives ptr_one's own slot in O(1) from its maintained
    // index, and .conjugated gives ptr_two's slot from there - no pool scan needed.
    VertexPointer * ptr_one_slot {cfg->internal_ph_manager->findPointer(ptr_one)};
    VertexPointer * ptr_two_slot {ptr_one_slot->conjugated};
    assert(ptr_two_slot != nullptr);
    cfg->internal_ph_manager->removeVertexPointers(*ptr_one_slot, *ptr_two_slot);

    // splice ptr_one and ptr_two out of the diagram and return them to the free pool.
    cfg->addVertexToPool(cfg->removeVertex(ptr_one));
    cfg->addVertexToPool(cfg->removeVertex(ptr_two));

    weight::LKMatrix::computeRightSide(cfg->diagram_head, last_survivor->next);
    weight::LKMatrix::computeLeftSide(cfg->diagram_tail, ptr_prev);

    sign.accepted(cfg);
}
