
#include "../../include/updates/rm_internal_ph.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <Eigen/Core>
#include "../../include/vertex_manager.hpp"
#include "../../include/weight_computation.hpp"

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

    int i {0};

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

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*ptr_prev->eff_masses[0])*(ptr_two->tau_next - ptr_prev->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*ptr_prev->eff_masses[1])*(ptr_two->tau_next - ptr_prev->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*ptr_prev->eff_masses[2])*(ptr_two->tau_next - ptr_prev->tau));

            proposed_weights.push_back(current_new_weight);
        }
        else if (ptr == ptr_one){
            current_new_weight.baseWF = ptr_prev->baseWF;
            current_new_weight.vertex_wf_component = Eigen::Matrix3d::Identity();
            
            p_fin = ptr_prev->k;
            current_new_weight.k = p_fin;

            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];

            current_new_weight.eff_masses = ptr_prev->eff_masses;

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*ptr_prev->eff_masses[0])*(ptr_one->tau_next - ptr_prev->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*ptr_prev->eff_masses[1])*(ptr_one->tau_next - ptr_prev->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*ptr_prev->eff_masses[2])*(ptr_one->tau_next - ptr_prev->tau));

            proposed_weights.push_back(current_new_weight);
        }
        else if (ptr->next == ptr_two){
            current_new_weight.baseWF = ptr_two->baseWF;
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_new_weight.baseWF);

            p_fin = ptr_two->k;
            current_new_weight.k = p_fin;

            p_fin_sq = p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2];
        
            current_new_weight.eff_masses = ptr_two->eff_masses;

            current_new_weight.el_prop_action(0,0) =  std::exp(-p_fin_sq/(2*ptr_two->eff_masses[0])*(ptr_two->tau_next - ptr_two->prev->tau));
            current_new_weight.el_prop_action(1,1) =  std::exp(-p_fin_sq/(2*ptr_two->eff_masses[1])*(ptr_two->tau_next - ptr_two->prev->tau));
            current_new_weight.el_prop_action(2,2) =  std::exp(-p_fin_sq/(2*ptr_two->eff_masses[2])*(ptr_two->tau_next - ptr_two->prev->tau));

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

            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights.push_back(current_new_weight);
        }

        ptr = ptr->next;
        ++i;
    } while (ptr != ptr_two);

    Eigen::Matrix3d new_matrix_product {Eigen::Matrix3d::Identity()};
    for(int j {i-1}; j > -1; --j){
        new_matrix_product = proposed_weights[j].vertex_wf_component * proposed_weights[j].el_prop_action * new_matrix_product;
    }

    const double weight_proposed {
        (new_matrix_product * ptr_two->next->vertex_wf_component * ptr_two->next->right_component * ptr_prev->left_component * ptr_prev->vertex_wf_component).trace()
    };

    const double weight_current {
        this->cfg->diagram_head->right_component.trace() *
        Coupling::Strength::compute(w_to_reject, ptr_one->ph_energy, ptr_one->diel_response) * 
            Coupling::Strength::compute(w_to_reject, ptr_one->ph_energy, ptr_one->diel_response) *
            std::exp(-ptr_one->ph_energy*(ptr_two->tau - ptr_one->tau))
    };

    const double p_A {static_cast<double>(this->cfg->internal_ph_manager->current_length)/2. - 1.};
    const double p_B {static_cast<double>(this->cfg->internal_ph_manager->current_length + this->cfg->external_ph_manager->current_length - 1)};

    const double numerator {
        p_A *
        weight_proposed *
        ptr_one->ph_energy * std::exp(-ptr_one->ph_energy*(tau_two - tau_one)) *
        std::pow(2.*std::numbers::pi, 3) *
        std::pow((tau_two - tau_one)/(2*std::numbers::pi), 1.5) *
        std::exp(-((w_to_reject[0]*w_to_reject[0]+w_to_reject[1]*w_to_reject[1]+w_to_reject[2]*w_to_reject[2])/2.)*(tau_two - tau_one))
    };

    const double tau_init_v1 {ptr_one->prev->tau};
    const double tau_end_v1 {ptr_one->next == ptr_two ? ptr_two->tau_next : ptr_one->tau_next};

    const double denominator {
        p_B *
        weight_current *
        (tau_end_v1 -tau_init_v1) *
        Coupling::Parameters::V_unit_cell
    };

    return numerator/denominator;
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
            ptr = ptr->next;
            ++idx;
        }
        ptr->k = proposed_weights.back().k;
        ptr->eff_masses = proposed_weights.back().eff_masses;
        ptr->baseWF = proposed_weights.back().baseWF;
        ptr->vertex_wf_component = proposed_weights.back().vertex_wf_component;
        ptr->el_prop_action = proposed_weights.back().el_prop_action;
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
}
