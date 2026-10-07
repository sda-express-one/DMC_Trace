#include "updates/chg_ph_energy.hpp"
#include <cmath>
#include <random>
#include "comp_method/vertex_coupling.hpp"
#include "diagram/phonon_manager.hpp"

double chg_ph_energy::attempt(){
    const int current_order {cfg->internal_ph_manager->current_length + cfg->external_ph_manager->current_length};

    // with a single mode there is nothing to change to
    if (current_order < 2 || cfg->phonon_mode_manager->num_phonon_modes < 2) {
        return -1.;
    }

    std::uniform_int_distribution<int> drawPtr{0, current_order/2-1};
    
    const int current_order_int {cfg->internal_ph_manager->current_length};
    const int ptr_index {drawPtr(*rng)};

    internal = ptr_index < current_order_int/2;
    if(internal){
        ptr_vertex = cfg->internal_ph_manager->selectVertex(2*ptr_index);
    }
    else {
        ptr_vertex = cfg->external_ph_manager->selectVertex(2*(ptr_index - cfg->internal_ph_manager->current_length/2));
    }
    
    ph_index = cfg->phonon_mode_manager->drawPhononMode();

    const double new_ph_energy {cfg->phonon_mode_manager->phonon_mode_pool[ph_index].phonon_energy};
    if(ptr_vertex->ph_energy == new_ph_energy){
        return -1;
    }

    const double new_diel_response {cfg->phonon_mode_manager->phonon_mode_pool[ph_index].diel_response};
    
    const double tau_one = ptr_vertex->tau;
    const double tau_two = ptr_vertex->conj_vertex->tau;
    // ptr_vertex is the creation vertex: tau_two - tau_one internal, wrapped through the boundary external
    const double length {internal ? tau_two - tau_one : cfg->current_tau_length - tau_one + tau_two};

    const double new_ph_action = std::exp(-new_ph_energy * length); 

    const double numerator {
        new_ph_action * Coupling::Strength::squaredTimesMomentumSquared(new_ph_energy, new_diel_response)
    };

    const double denominator {
        ptr_vertex->ph_action * Coupling::Strength::squaredTimesMomentumSquared(ptr_vertex->ph_energy, ptr_vertex->diel_response)
    };

    return sign.take(numerator / denominator);
}

void chg_ph_energy::accept(){
    const PhononMode & mode {cfg->phonon_mode_manager->phonon_mode_pool[ph_index]};
    Vertex * conj {ptr_vertex->conj_vertex};

    // both ends of the line carry the mode
    ptr_vertex->ph_energy = mode.phonon_energy;
    ptr_vertex->diel_response = mode.diel_response;
    conj->ph_energy = mode.phonon_energy;
    conj->diel_response = mode.diel_response;

    // propagator (sets both ends) and the cached couplings, which depend on omega and epsilon
    internal ? ptr_vertex->computeInternalPhPropAction() : ptr_vertex->computeExternalPhPropAction(cfg->current_tau_length);
    ptr_vertex->vertexStrength();
    conj->vertexStrength();

    sign.accepted(cfg);
}
