#include "../../include/updates/swp_ph.hpp"
#include "../../include/diagram/vertex_manager.hpp"
#include "../../include/comp_method/weight_computation.hpp"
#include <array>
#include <utility>

double swp_ph_update::attempt(){
    if(cfg->internal_ph_manager->current_length < 4){
        return -1.;
    }

    ptr_one = cfg->internal_ph_manager->chooseAnyVertex()->linked_vertex;
    assert(ptr_one != nullptr);

    ptr_two = ptr_one->next;
    assert(ptr_two != nullptr);

    // reject immediately unless ptr_two is also an internal-line endpoint (|type| == 1)
    if(std::abs(ptr_two->type) % 2 == 0){
        return -1.;
    }

    // reject if they're already each other's conjugate (a short line with nothing in between) -
    // "swapping" their links in that case would make a vertex its own conjugate.
    if(ptr_one->conj_vertex == ptr_two){
        return -1.;
    }

    // ... swap logic once both are confirmed internal
    const int c_one {ptr_one->type};
    const std::array<double, 3> w_one {ptr_one->w};
    const double tau_one {ptr_one->tau};
    const double ph_energy_one {ptr_one->ph_energy};
    
    const int c_two {ptr_two->type};
    const std::array<double, 3> w_two {ptr_two->w};
    const double tau_two {ptr_two->tau};
    const double ph_energy_two {ptr_two->ph_energy};

    const std::array<double, 3> k_between {ptr_one->k};
    k_new = {
        k_between[0] + c_one*w_one[0] - c_two*w_two[0], 
        k_between[1] + c_one*w_one[1] - c_two*w_two[1],
        k_between[2] + c_one*w_one[2] - c_two*w_two[2]
    };
    const Eigen::Matrix<double, 4, 3> eigensolution_wrapper {weight::LKMatrix::diagonalizeLKHamiltonian(k_new)};
    const std::array<double, 3> eigenvalues {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};

    proposed_weights[0].eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
    proposed_weights[0].baseWF = eigensolution_wrapper.block<3,3>(1,0);
    proposed_weights[0].vertex_wf_component = Coupling::LKOverlap::computeMatrix(ptr_one->prev->baseWF, proposed_weights[0].baseWF);
    
    const double k_new_sq {k_new[0]*k_new[0] + k_new[1]*k_new[1] + k_new[2]*k_new[2]};

    proposed_weights[0].el_prop_action(0,0) = std::exp(-k_new_sq/(2*proposed_weights[0].eff_masses[0])*(tau_two - tau_one));
    proposed_weights[0].el_prop_action(1,1) = std::exp(-k_new_sq/(2*proposed_weights[0].eff_masses[1])*(tau_two - tau_one));
    proposed_weights[0].el_prop_action(2,2) = std::exp(-k_new_sq/(2*proposed_weights[0].eff_masses[2])*(tau_two - tau_one));

    //proposed_weights[1].eff_masses = ptr_two->eff_masses;
    proposed_weights[1].baseWF = ptr_two->baseWF;
    proposed_weights[1].vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights[0].baseWF, proposed_weights[1].baseWF);
    
    //proposed_weights[1].el_prop_action = ptr_two->el_prop_action;

    const double ph_energy_tot {c_one*ph_energy_one - c_two*ph_energy_two};

    const double numerator {
        (
         proposed_weights[0].vertex_wf_component *
         proposed_weights[0].el_prop_action *
         proposed_weights[1].vertex_wf_component * 
         ptr_two->right_component * 
         ptr_one->left_component
        ).trace() *
        std::exp(-ph_energy_tot*(tau_two - tau_one))
    };

    const double denominator{(this->cfg->diagram_head->right_component).trace()};

    return numerator/denominator;
}

void swp_ph_update::accept(){
    // ptr_one's own outgoing segment [tau_one, tau_two] takes the newly-computed (k_new-derived)
    // state staged in attempt().
    ptr_one->k = k_new;
    ptr_one->eff_masses = proposed_weights[0].eff_masses;
    ptr_one->baseWF = proposed_weights[0].baseWF;
    ptr_one->vertex_wf_component = proposed_weights[0].vertex_wf_component;
    ptr_one->el_prop_action = proposed_weights[0].el_prop_action;

    // ptr_two's own outgoing segment is untouched (same momentum, same duration) - only its
    // incoming overlap changes, since what feeds into it (ptr_one's new baseWF) changed.
    ptr_two->vertex_wf_component = proposed_weights[1].vertex_wf_component;

    // Swap the two vertices' phonon-line identities - tau/tau_next/prev/next stay fixed, only
    // which line each is attached to changes. The far-side conjugates must be re-pointed too,
    // or the bidirectional conj_vertex invariant breaks.
    Vertex * distant_one {ptr_one->conj_vertex};
    Vertex * distant_two {ptr_two->conj_vertex};

    std::swap(ptr_one->type, ptr_two->type);
    std::swap(ptr_one->w, ptr_two->w);
    std::swap(ptr_one->ph_energy, ptr_two->ph_energy);
    std::swap(ptr_one->diel_response, ptr_two->diel_response);

    ptr_one->conj_vertex = distant_two;
    ptr_two->conj_vertex = distant_one;
    distant_one->conj_vertex = ptr_two;
    distant_two->conj_vertex = ptr_one;

    // the VertexPointer pool tracks the same conjugate relationship in parallel via .conjugated -
    // it needs the identical four-way fix-up, or the pool's own bookkeeping goes stale even
    // though Vertex::conj_vertex above is now correct.
    VertexPointer * slot_one {cfg->internal_ph_manager->findPointer(ptr_one)};
    VertexPointer * slot_two {cfg->internal_ph_manager->findPointer(ptr_two)};
    VertexPointer * slot_distant_one {cfg->internal_ph_manager->findPointer(distant_one)};
    VertexPointer * slot_distant_two {cfg->internal_ph_manager->findPointer(distant_two)};

    slot_one->conjugated = slot_distant_two;
    slot_two->conjugated = slot_distant_one;
    slot_distant_one->conjugated = slot_two;
    slot_distant_two->conjugated = slot_one;

    // vertex_strength_component depends on w/ph_energy/diel_response/eff_masses, all of which
    // changed on at least one of the two vertices - refresh both.
    ptr_one->vertexStrength();
    ptr_two->vertexStrength();

    // each vertex is now attached to a different line; recomputing on ptr_one/ptr_two also
    // refreshes distant_two/distant_one (the other end of each line) via conj_vertex.
    ptr_one->computeInternalPhPropAction();
    ptr_two->computeInternalPhPropAction();

    weight::LKMatrix::computeRightSide(cfg->diagram_head, ptr_two->next);
    weight::LKMatrix::computeLeftSide(cfg->diagram_tail, ptr_one);
}
