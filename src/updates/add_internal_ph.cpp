#include "../../include/updates/add_internal_ph.hpp"
#include <array>
#include <cmath>
#include <pthread.h>
#include <random>
#include "../../include/weight_computation.hpp"
#include "../../include/phonon_manager.hpp"

double add_int_ph_update::attempt(){
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
    
    // to be implemented
    const int ph_index {0};

    this->tau_one = std_unif(*rng)*(tau_end_v1 - tau_init_v1);

    const int ph_mode_index {this->cfg->phonon_mode_manager->drawPhononMode()};
    const double ph_mode_energy {this->cfg->phonon_mode_manager->phonon_mode_pool[ph_mode_index].phonon_energy};
    const double ph_mode_diel_response {this->cfg->phonon_mode_manager->phonon_mode_pool[ph_mode_index].diel_response};

    this->tau_two = this->tau_one - std::log(1 - std_unif(*rng))/ph_mode_energy;

    if(tau_two > this->cfg->tau_max){
        return -1;
    }
    
    
    std::normal_distribution<double> distrib_norm(0, std::sqrt(1/(tau_two - tau_one)));
    std::array<double, 3> w_values_new {distrib_norm(*this->rng), distrib_norm(*this->rng), distrib_norm(*this->rng)};

    // find position of new tau values
    ptr_two = this->cfg->findPositionFromLeft(ptr_one, this->tau_two);
    
    std::array<double, 3> p_init {0., 0., 0.};
    std::array<double, 3> p_fin {0., 0., 0.};

    double energy_init {0.};
    double energy_fin {0.};

    Vertex * ptr {ptr_one};
    
    do {
        p_init = ptr->k;
        p_fin = {ptr->k[0] - w_values_new[0], ptr->k[1] - w_values_new[1], ptr->k[2] - w_values_new[2]};


        ptr = ptr->next;
    } while (ptr != ptr_two->next);
};
