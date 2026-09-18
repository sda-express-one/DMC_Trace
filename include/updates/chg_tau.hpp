#ifndef CHG_TAU_HPP
#define CHG_TAU_HPP

#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include "../diagram/vertex.hpp"
#include "../comp_method/weight_computation.hpp"
#include "../diagram/diagram_config.hpp"

struct chg_tau_update {
    diagram_cfg * cfg;
    simplemc::xoshiro256ss* rng;
    mutable std::uniform_real_distribution<double> std_unif {0.,1.};
    Vertex * vertex {nullptr};
    Eigen::Matrix3d new_action {Eigen::Matrix3d::Identity()};
    double tau_last_vertex {0.};
    double tau_proposed {0.};

    double attempt(){
        this->vertex = this->cfg->diagram_tail->prev;
        assert(this->vertex != nullptr);
        this->tau_last_vertex = this->vertex->tau;
        
        const std::array<double, 3> energies {this->vertex->electronEnergy()};
        const double lowest_energy {*std::min_element(energies.begin(), energies.end())};
        double ext_ph_energies {0.};
        
        {
            int counter {0};

            for(int i {0}; i < this->cfg->external_ph_manager->current_length; ++i){
                if(this->cfg->external_ph_manager->ptr_vertex_pool[i].linked_vertex->type == -2){
                    ext_ph_energies += this->cfg->external_ph_manager->ptr_vertex_pool[i].linked_vertex->ph_energy;
                    ++counter;
                }
                if(counter == this->cfg->external_ph_manager->current_length/2){
                    break;
                }
            }
        }

        const double total_lowest_energy {lowest_energy - this->cfg->chem_pot + ext_ph_energies};

        tau_proposed = this->tau_last_vertex - std::log(1 - std_unif(*rng))/total_lowest_energy;

        if(tau_proposed > this->cfg->tau_max){
            return -1.;
        }

        this->new_action(0,0) = std::exp(-energies[0]*(tau_proposed - this->tau_last_vertex));
        this->new_action(1,1) = std::exp(-energies[1]*(tau_proposed - this->tau_last_vertex));
        this->new_action(2,2) = std::exp(-energies[2]*(tau_proposed - this->tau_last_vertex));

        const double diagram_weight_current {
            (vertex->vertex_wf_component * vertex->el_prop_action.diagonal().asDiagonal() * vertex->left_component).trace()
        };
        const double diagram_weight_proposed {
            (vertex->vertex_wf_component * this->new_action * vertex->left_component).trace()
        };

        double numerator {std::exp(-lowest_energy * this->cfg->diagram_tail->tau) * diagram_weight_proposed};
        double denominator {std::exp(-lowest_energy * tau_proposed) * diagram_weight_current};

        return (numerator/denominator);
    }

    void accept(){
        this->vertex->tau_next = tau_proposed;
        this->cfg->diagram_tail->tau = tau_proposed;
        this->cfg->current_tau_length = tau_proposed;

        this->vertex->el_prop_action = this->new_action;

        weight::LKMatrix::computeRightSide(cfg->diagram_head, cfg->diagram_tail);

        {
            int counter {0};

            for(int i {0}; i < this->cfg->external_ph_manager->current_length; ++i){
                if(this->cfg->external_ph_manager->ptr_vertex_pool[i].linked_vertex->type == -2){
                    this->cfg->external_ph_manager->ptr_vertex_pool[i].linked_vertex->computeExternalPhPropAction(tau_proposed);
                    ++counter;
                }
                if(counter == this->cfg->external_ph_manager->current_length/2){
                    break;
                }
            }
        }
    }
};

#endif // !CHG_TAU_HPP
