#include "updates/chg_ph_momentum.hpp"
#include <array>
#include <cmath>
#include <random>
#include "comp_method/vertex_coupling.hpp"
#include "comp_method/weight_computation.hpp"
#include "diagram/vertex.hpp"

double chg_ph_momentum::attempt(){
    const int current_order {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};
        
    std::uniform_int_distribution<int> choose_ph_vertex {0, current_order - 1};
    int chosen_vertex {choose_ph_vertex(*rng)};

    double k_new_sq {0.};
    Eigen::Matrix<double, 4, 3> eigensolution_wrapper;
    std::array<double, 3> eigenvalues {1., 1., 1.};
        
    if (chosen_vertex < cfg->internal_ph_manager->current_length) {
        ptr_one = cfg->internal_ph_manager->selectVertex(chosen_vertex);
            
        if (ptr_one->type != 1) {
            ptr_two = ptr_one;
            ptr_one = ptr_two->conj_vertex;
        }
        else {
            ptr_two = ptr_one->conj_vertex;
        }

        const double tau_one {ptr_one->tau};
        const double tau_two {ptr_two->tau};
            
        std::normal_distribution<double> distrib_norm(0, std::sqrt(1/(tau_two - tau_one)));

        w_proposed = {distrib_norm(*rng), distrib_norm(*rng), distrib_norm(*rng)};

        const std::array<double, 3> w_current {ptr_one->w};

        Vertex * ptr {ptr_one};
        int i {0};

        while(ptr != ptr_two){
            weight::ProposedVertexWeight current_new_weight;
            current_new_weight.k[0] = ptr->k[0] + w_current[0] - w_proposed[0];
            current_new_weight.k[1] = ptr->k[1] + w_current[1] - w_proposed[1];
            current_new_weight.k[2] = ptr->k[2] + w_current[2] - w_proposed[2];

            k_new_sq = current_new_weight.k[0]*current_new_weight.k[0] + current_new_weight.k[1]*current_new_weight.k[1] + current_new_weight.k[2]*current_new_weight.k[2];

            eigensolution_wrapper = weight::LKMatrix::diagonalizeLKHamiltonian(current_new_weight.k);

            eigenvalues = {eigensolution_wrapper(0,0), eigensolution_wrapper(0,1), eigensolution_wrapper(0,2)};
            current_new_weight.eff_masses = weight::LKMatrix::computeEffMassfromEigenval(eigenvalues);
            current_new_weight.baseWF = eigensolution_wrapper.block<3,3>(1,0);
            
            if(ptr == ptr_one){
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(ptr_one->prev->baseWF, current_new_weight.baseWF);
            }
            else {
                current_new_weight.vertex_wf_component = Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, current_new_weight.baseWF);
            }

            current_new_weight.el_prop_action(0,0) = std::exp(-k_new_sq/(2*current_new_weight.eff_masses[0])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(1,1) = std::exp(-k_new_sq/(2*current_new_weight.eff_masses[1])*(ptr->tau_next - ptr->tau));
            current_new_weight.el_prop_action(2,2) = std::exp(-k_new_sq/(2*current_new_weight.eff_masses[2])*(ptr->tau_next - ptr->tau));

            proposed_weights.push_back(current_new_weight);
            
            ++i;
            ptr = ptr->next;
        }

        Eigen::Matrix3d new_matrix_product {Coupling::LKOverlap::computeMatrix(proposed_weights.back().baseWF, ptr_two->baseWF)};
        for(int j {i-1}; j > -1; --j){
            new_matrix_product = proposed_weights[j].vertex_wf_component * proposed_weights[j].el_prop_action * new_matrix_product;
        }

        const double ph_energy {ptr_one->ph_energy};
        const double ph_diel_response {ptr_one->diel_response};

        const double weights_proposed {
            (new_matrix_product * ptr_two->right_component * ptr_one->left_component).trace() *
            Coupling::Strength::compute(w_proposed, ph_energy, ph_diel_response) *
            Coupling::Strength::compute(w_proposed, ph_energy, ph_diel_response)
        };

        const double weights_current {
            cfg->diagram_head->right_component.trace() *
            Coupling::Strength::compute(w_current, ph_energy, ph_diel_response) *
            Coupling::Strength::compute(w_current, ph_energy, ph_diel_response)
        };

        const double numerator {
            weights_proposed *
            std::exp(-(w_current[0]*w_current[0] + w_current[1]*w_current[1] + w_current[2]*w_current[2])/2.)
        };

        const double denominator {
            weights_current * 
            std::exp(-(w_proposed[0]*w_proposed[0] + w_proposed[1]*w_proposed[1] + w_proposed[2]*w_proposed[2])/2.)
        };

        return numerator/denominator;
    }
    else {
        chosen_vertex -= cfg->internal_ph_manager->current_length;
        ptr_one = cfg->external_ph_manager->selectVertex(chosen_vertex);

        if(ptr_one->type != -2){
            ptr_two = ptr_one;
            ptr_one = ptr_two->conj_vertex;
        }
        else {
            ptr_two = ptr_one->conj_vertex;
        }

        
    }
}
