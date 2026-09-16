#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include "../../include/updates/add_internal_ph.hpp"
#include "../../include/vertex_coupling.hpp"
#include "../../include/weight_computation.hpp"
#include "../../include/phonon_manager.hpp"

double add_int_ph_update::attempt(){
    // discard whatever the previous attempt() staged here - keeps the reserved capacity
    // from the constructor intact, just resets size() to 0.
    proposed_weights.clear();

    if(cfg->internal_ph_manager->current_length == cfg->internal_ph_manager->max_length){
        return -1;
    }

    const int current_total_order {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};

    // choose random electron propagator for new vertex
    std::uniform_int_distribution<int> choose_prop_el {0, current_total_order};
    int chosen_prop {choose_prop_el(*this->rng)};

    if(chosen_prop < this->cfg->internal_ph_manager->current_length){
        ptr_one = this->cfg->internal_ph_manager->selectVertex(chosen_prop);
    }
    else if(chosen_prop < current_total_order){
        chosen_prop -= this->cfg->internal_ph_manager->current_length;
        ptr_one = this->cfg->external_ph_manager->selectVertex(chosen_prop);
    }
    else {
        chosen_prop = 0;
        ptr_one = cfg->diagram_head;
    }

    const double tau_init_v1 {ptr_one->tau};
    const double tau_end_v1 {ptr_one->tau_next};
    
    this->tau_one = tau_init_v1 + std_unif(*rng)*(tau_end_v1 - tau_init_v1);

    ph_index = this->cfg->phonon_mode_manager->drawPhononMode();
    const double ph_mode_energy {this->cfg->phonon_mode_manager->phonon_mode_pool[ph_index].phonon_energy};
    const double ph_mode_diel_response {this->cfg->phonon_mode_manager->phonon_mode_pool[ph_index].diel_response};

    this->tau_two = this->tau_one - std::log(1 - std_unif(*rng))/ph_mode_energy;

    // bounded against the diagram's current extent, not the fixed tau_max ceiling: tau_max only
    // bounds where diagram_tail is itself allowed to reach (enforced in chg_tau_update), and
    // diagram_tail->tau is frequently much smaller than that - a phonon vertex can't legally sit
    // beyond where the worldline currently ends, and findPositionFromLeft would walk off the end
    // of the diagram (UB in release builds) if it tried.
    if(tau_two > this->cfg->diagram_tail->tau){
        return -1;
    }
    
    
    std::normal_distribution<double> distrib_norm(0, std::sqrt(1/(tau_two - tau_one)));
    this->w_proposed = {distrib_norm(*this->rng), distrib_norm(*this->rng), distrib_norm(*this->rng)};

    // find position of new tau values
    ptr_two = this->cfg->findPositionFromLeft(ptr_one, this->tau_two);
    
    std::array<double, 3> p_fin {0., 0., 0.};
    double p_fin_sq {0.};

    Vertex * ptr {ptr_one};

    int i {0};
    
    Eigen::Matrix<double, 4, 3> eigensolution_wrapper;
    std::array<double, 3> eigenvalues {1., 1., 1.};
    
    do {
        weight::ProposedVertexWeight current_new_weight;
        p_fin = {ptr->k[0] - w_proposed[0], ptr->k[1] - w_proposed[1], ptr->k[2] - w_proposed[2]};
        current_new_weight.k = p_fin;
        p_fin_sq = {p_fin[0]*p_fin[0] + p_fin[1]*p_fin[1] + p_fin[2]*p_fin[2]};

        eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(p_fin);
        
        eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
        current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);

        if(ptr_one == ptr_two){
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(ptr_one->baseWF, current_new_weight.baseWF);  
            
            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(this->tau_two - this->tau_one)); 
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(this->tau_two - this->tau_one));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(this->tau_two - this->tau_one));
            
            weight::ProposedVertexWeight second_current_new_weight;

            second_current_new_weight.k = ptr_two->k;
            second_current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(current_new_weight.baseWF, ptr_two->baseWF);
            second_current_new_weight.eff_masses = ptr_two->eff_masses;

            double k_final_sq = second_current_new_weight.k[0]*second_current_new_weight.k[0] + second_current_new_weight.k[1]*second_current_new_weight.k[1] + second_current_new_weight.k[2]*second_current_new_weight.k[2];
            second_current_new_weight.el_prop_action(0,0) = std::exp(-k_final_sq/(2*second_current_new_weight.eff_masses[0])*(ptr_two->tau_next - tau_two));
            second_current_new_weight.el_prop_action(1,1) = std::exp(-k_final_sq/(2*second_current_new_weight.eff_masses[1])*(ptr_two->tau_next - tau_two));
            second_current_new_weight.el_prop_action(2,2) = std::exp(-k_final_sq/(2*second_current_new_weight.eff_masses[2])*(ptr_two->tau_next - tau_two));

            proposed_weights.push_back(current_new_weight);
            proposed_weights.push_back(second_current_new_weight);
        }
        else if(ptr == ptr_one){
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(ptr_one->baseWF, current_new_weight.baseWF);
            
            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr_one->tau_next - this->tau_one)); 
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr_one->tau_next - this->tau_one));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr_one->tau_next - this->tau_one));

            proposed_weights.push_back(current_new_weight);
        }
        else if(ptr == ptr_two){
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights[i-1].baseWF, current_new_weight.baseWF);
            
            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(this->tau_two - ptr_two->tau)); 
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(this->tau_two - ptr_two->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(this->tau_two - ptr_two->tau));

            weight::ProposedVertexWeight second_current_new_weight;

            second_current_new_weight.k = ptr_two->k;
            second_current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(current_new_weight.baseWF, ptr_two->baseWF);
            second_current_new_weight.eff_masses = ptr_two->eff_masses;
            double k_final_sq = second_current_new_weight.k[0]*second_current_new_weight.k[0] + second_current_new_weight.k[1]*second_current_new_weight.k[1] + second_current_new_weight.k[2]*second_current_new_weight.k[2];
            second_current_new_weight.el_prop_action(0,0) = std::exp(-k_final_sq/(2*second_current_new_weight.eff_masses[0])*(ptr_two->tau_next - tau_two));
            second_current_new_weight.el_prop_action(1,1) = std::exp(-k_final_sq/(2*second_current_new_weight.eff_masses[1])*(ptr_two->tau_next - tau_two));
            second_current_new_weight.el_prop_action(2,2) = std::exp(-k_final_sq/(2*second_current_new_weight.eff_masses[2])*(ptr_two->tau_next - tau_two));

            proposed_weights.push_back(current_new_weight);
            proposed_weights.push_back(second_current_new_weight);
        }
        else{
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights[i-1].baseWF, current_new_weight.baseWF);
            
            current_new_weight.el_prop_action(0,0) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau)); 
            current_new_weight.el_prop_action(1,1) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-p_fin_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights.push_back(current_new_weight);
        }

        ptr = ptr->next;
        ++i;
    } while (ptr != ptr_two->next);

    // ptr_one keeps its own identity (unlike ptr_two's other end in rm_internal_ph, nothing here
    // merges it away), but its own leading segment shrinks from [tau_init_v1, tau_end_v1] down to
    // [tau_init_v1, tau_one] - same (unshifted) momentum, shorter duration, so its cached action is
    // now stale and has to be recomputed rather than reused or left out of the trace.
    const std::array<double, 3> ptr_one_energies {ptr_one->electronEnergy()};
    Eigen::Matrix3d ptr_one_action_new {Eigen::Matrix3d::Identity()};
    ptr_one_action_new(0,0) = std::exp(-ptr_one_energies[0]*(this->tau_one - tau_init_v1));
    ptr_one_action_new(1,1) = std::exp(-ptr_one_energies[1]*(this->tau_one - tau_init_v1));
    ptr_one_action_new(2,2) = std::exp(-ptr_one_energies[2]*(this->tau_one - tau_init_v1));

    // fold every proposed_weights entry's wf*action (including the last one's action, unlike
    // before) - the chain then needs to be closed out on the right by ptr_two->next (skipping
    // ptr_two's own now-superseded right_component, whose leading factor is its stale action)
    // and on the left by ptr_one's own (unchanged wf, newly-recomputed action) leading segment.
    Eigen::Matrix3d new_matrix_product {Eigen::Matrix3d::Identity()};
    for(int j {i}; j > -1; --j){
        new_matrix_product = proposed_weights[j].vertex_wf_component * proposed_weights[j].el_prop_action * new_matrix_product;
    }

    const double weights_proposed {
        (new_matrix_product * ptr_two->next->vertex_wf_component * ptr_two->next->right_component * ptr_one->left_component * ptr_one->vertex_wf_component * ptr_one_action_new).trace() *
            Coupling::Strength::compute(w_proposed, ph_mode_energy, ph_mode_diel_response) *
            Coupling::Strength::compute(w_proposed, ph_mode_energy, ph_mode_diel_response) *
            std::exp(-ph_mode_energy*(this->tau_two - this->tau_one))
    };
    const double weights_current {this->cfg->diagram_head->right_component.trace()};

    const double p_B {static_cast<double>(this->cfg->internal_ph_manager->current_length + this->cfg->external_ph_manager->current_length + 1)};
    const double p_A {static_cast<double>(this->cfg->internal_ph_manager->current_length)/2. + 1.};
    
    // add context factors
    const double numerator {
        p_B *
        weights_proposed * 
        (tau_end_v1 - tau_init_v1) * 
        Coupling::Parameters::V_unit_cell
    };
    const double denominator {
        p_A *
        weights_current *
        std::pow(2.*std::numbers::pi, 3) * 
        std::pow((this->tau_two - this->tau_one)/(2*std::numbers::pi), 1.5) * 
        std::exp(-((w_proposed[0]*w_proposed[0]+w_proposed[1]*w_proposed[1]+w_proposed[2]*w_proposed[2])/2.)*(this->tau_two - this->tau_one))
    };

    return numerator/denominator;
};

void add_int_ph_update::accept(){
    const double ph_mode_energy {cfg->phonon_mode_manager->phonon_mode_pool[ph_index].phonon_energy};
    const double ph_mode_diel_response {cfg->phonon_mode_manager->phonon_mode_pool[ph_index].diel_response};

    // 1. Draw the two new vertices before touching any linkage, so nothing here can leave the
    //    diagram half-mutated if something's wrong.
    Vertex * v_one {cfg->drawVertexFromPool()};
    Vertex * v_two {cfg->drawVertexFromPool()};

    // 2. Commit the shifted-momentum state onto every EXISTING vertex the walk touched, using the
    //    still-intact ptr_one->...->ptr_two chain (nothing spliced yet). proposed_weights.front()
    //    belongs to v_one (new) and proposed_weights.back() belongs to v_two (new) - only the
    //    entries strictly between them are existing interior vertices, plus ptr_two itself when it
    //    differs from ptr_one.
    if (ptr_one != ptr_two) {
        Vertex * ptr {ptr_one->next};
        std::size_t idx {1};
        while (ptr != ptr_two) {
            ptr->k = proposed_weights[idx].k;
            ptr->eff_masses = proposed_weights[idx].eff_masses;
            ptr->baseWF = proposed_weights[idx].baseWF;
            ptr->vertex_wf_component = proposed_weights[idx].vertex_wf_component;
            ptr->el_prop_action = proposed_weights[idx].el_prop_action;
            ptr = ptr->next;
            ++idx;
        }
        // ptr is now ptr_two; proposed_weights[idx] (== .size()-2) is its own shifted, shortened segment
        ptr_two->k = proposed_weights[idx].k;
        ptr_two->eff_masses = proposed_weights[idx].eff_masses;
        ptr_two->baseWF = proposed_weights[idx].baseWF;
        ptr_two->vertex_wf_component = proposed_weights[idx].vertex_wf_component;
        ptr_two->el_prop_action = proposed_weights[idx].el_prop_action;
        ptr_two->tau_next = tau_two;
    }

    // ptr_one's own leading segment is untouched by the phonon shift either way - only its length,
    // and hence its cached el_prop_action, changes (same momentum, shorter duration).
    ptr_one->tau_next = tau_one;
    ptr_one->computeElPropAction();

    // 3. Splice v_one and v_two into the linked list. If ptr_one == ptr_two, v_two must go after
    //    v_one (not after ptr_two again, which is the same node).
    cfg->addVertex(v_one, ptr_one);
    cfg->addVertex(v_two, (ptr_one == ptr_two) ? v_one : ptr_two);

    // 4. Fill in the new vertices' own state from the first/last proposed_weights entries.
    // Both tau's are set before either tau_next is derived from ->next->tau: when
    // ptr_one == ptr_two, v_one->next is v_two itself, so v_two->tau must already be
    // correct by the time v_one->tau_next reads it.
    v_one->tau = tau_one;
    v_two->tau = tau_two;

    v_one->tau_next = v_one->next->tau;
    v_one->type = +1; // creation - earlier in tau, per the internal-line convention
    v_one->k = proposed_weights.front().k;
    v_one->w = w_proposed;
    v_one->ph_energy = ph_mode_energy;
    v_one->diel_response = ph_mode_diel_response;
    v_one->eff_masses = proposed_weights.front().eff_masses;
    v_one->baseWF = proposed_weights.front().baseWF;
    v_one->vertex_wf_component = proposed_weights.front().vertex_wf_component;
    v_one->el_prop_action = proposed_weights.front().el_prop_action;

    v_two->tau_next = v_two->next->tau;
    v_two->type = -1; // annihilation - later in tau
    v_two->k = proposed_weights.back().k;
    v_two->w = w_proposed;
    v_two->ph_energy = ph_mode_energy;
    v_two->diel_response = ph_mode_diel_response;
    v_two->eff_masses = proposed_weights.back().eff_masses;
    v_two->baseWF = proposed_weights.back().baseWF;
    v_two->vertex_wf_component = proposed_weights.back().vertex_wf_component;
    v_two->el_prop_action = proposed_weights.back().el_prop_action;

    v_one->conj_vertex = v_two;
    v_two->conj_vertex = v_one;

    // vertex_strength_component depends only on each vertex's own w/ph_energy/diel_response/eff_masses,
    // all of which are now set - so it only needs computing on the two new vertices.
    v_one->vertexStrength();
    v_two->vertexStrength();

    // 5. Register the new line and refresh cached phonon/electron weights.
    cfg->internal_ph_manager->addVertexPointers(v_one, v_two);
    v_one->computeInternalPhPropAction(); // sets ph_action on both v_one and v_two (conj_vertex)

    weight::LKMatrix::computeRightSide(cfg->diagram_head, v_two->next);
    weight::LKMatrix::computeLeftSide(cfg->diagram_tail, v_one->prev);
}

