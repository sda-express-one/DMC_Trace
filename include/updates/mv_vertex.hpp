#ifndef MV_VERTEX_HPP
#define MV_VERTEX_HPP

#include <cassert>
#include <random>
#include <simplemc/random/xoshiro256.hpp>
#include "../vertex.hpp"
#include "updates_config.hpp"
#include "../weight_computation.hpp"

struct mv_tau_update {
    updates_cfg * cfg;
    simplemc::xoshiro256ss* rng;
    mutable std::uniform_real_distribution<double> std_unif {0.,1.};
    Vertex * vertex {nullptr};
    Eigen::Matrix3d new_action_el_incoming {Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d new_action_el_outgoing {Eigen::Matrix3d::Identity()};
    int index {-1};
    double tau_proposed {0.};

    double attempt(){
        const int total_current_order {this->cfg->internal_ph_manager->current_length + this->cfg->external_ph_manager->current_length};
        if(total_current_order < 1){
            return -1.;
        }

        std::uniform_int_distribution<int> choose_v {0, total_current_order - 1};
        index = choose_v(*rng); 

        if(index < this->cfg->internal_ph_manager->current_length){
            vertex = this->cfg->internal_ph_manager->selectVertex(index);
        }
        else {
            index -= this->cfg->internal_ph_manager->current_length;
            vertex = this->cfg->external_ph_manager->selectVertex(index);
        }
        assert(vertex != nullptr);
        assert(vertex != this->cfg->diagram_head);
        assert(vertex != this->cfg->diagram_tail);
        assert(vertex->prev != nullptr);
        assert(vertex->next != nullptr);
        
        const double tau_prev {vertex->prev->tau};
        const double tau_next {vertex->tau_next};

        const std::array<double, 3> energies_incoming {this->vertex->prev->electronEnergy()};
        const double lowest_energy_incoming {*std::min_element(energies_incoming.begin(), energies_incoming.end())};

        const std::array<double, 3> energies_outgoing {this->vertex->electronEnergy()};
        const double lowest_energy_outgoing {*std::min_element(energies_outgoing.begin(), energies_outgoing.end())};

        const int ph_type {vertex->type > 0 ? +1 : -1};

        const double ph_energy {vertex->phononEnergy()*static_cast<double>(ph_type)};

        const double deltaE {lowest_energy_incoming - lowest_energy_outgoing - ph_energy}; 

        tau_proposed = tau_prev - std::log(1 - std_unif(*rng)*(1 - std::exp(-deltaE*(tau_next - tau_prev))))/deltaE;

        assert(tau_proposed > tau_prev);
        assert(tau_proposed < tau_next);

        const double diagram_weight_current {
            (
                vertex->vertex_wf_component *
                vertex->right_component *
                vertex->left_component
            ).trace()
        };
        
        new_action_el_incoming(0,0) = std::exp(-energies_incoming[0]*(tau_proposed-tau_prev));
        new_action_el_incoming(1,1) = std::exp(-energies_incoming[1]*(tau_proposed-tau_prev));
        new_action_el_incoming(2,2) = std::exp(-energies_incoming[2]*(tau_proposed-tau_prev));

        new_action_el_outgoing(0,0) = std::exp(-energies_outgoing[0]*(tau_next-tau_proposed));
        new_action_el_outgoing(1,1) = std::exp(-energies_outgoing[1]*(tau_next-tau_proposed));
        new_action_el_outgoing(2,2) = std::exp(-energies_outgoing[2]*(tau_next-tau_proposed));

        const double diagram_weight_proposed {
            (
                vertex->prev->vertex_wf_component *
                new_action_el_incoming *
                vertex->vertex_wf_component *
                new_action_el_outgoing *
                vertex->next->vertex_wf_component *
                vertex->next->right_component *
                vertex->prev->left_component
            ).trace() 
        };

        const double numerator {std::exp(-deltaE * vertex->tau) * diagram_weight_proposed * std::exp(-ph_energy*(this->vertex->tau - this->tau_proposed))};
        const double denominator {std::exp(-deltaE * tau_proposed) * diagram_weight_current};

        return (numerator/denominator);
    }

    void accept(){
        this->vertex->prev->tau_next = tau_proposed;
        this->vertex->tau = tau_proposed;
        
        this->vertex->prev->el_prop_action = new_action_el_incoming;
        this->vertex->el_prop_action =  new_action_el_outgoing;

        weight::LKMatrix::computeRightSide(this->cfg->diagram_head, this->vertex->next);
        weight::LKMatrix::computeLeftSide(this->cfg->diagram_tail, this->vertex->prev);

        std::abs(this->vertex->type) % 2 == 1 ? this->vertex->computeInternalPhPropAction() : this->vertex->computeExternalPhPropAction(this->cfg->diagram_tail->tau);
    }
};

#endif // !MV_VERTEX_HPP

