#ifndef CHG_TAU_HPP
#define CHG_TAU_HPP

#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include "../vertex.hpp"
#include "../weight_computation.hpp"

struct chg_tau_cfg {
    Vertex * diagram_head {nullptr};
    Vertex * diagram_tail {nullptr};
    
    const double tau_max {50.};
    const double chem_pot {-1.};

    chg_tau_cfg(Vertex * diagram_head, Vertex * diagram_tail, double tau_max = 50.0, double chem_pot = -1.0) 
        : diagram_head(diagram_head), diagram_tail(diagram_tail), tau_max(tau_max), chem_pot(chem_pot) {
        assert(this->diagram_tail != nullptr);
        assert(this->chem_pot < 0);     
    }
};

struct chg_tau_update {
    chg_tau_cfg* cfg;
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
        tau_proposed = this->tau_last_vertex - std::log(1 - std_unif(*rng))/(lowest_energy - this->cfg->chem_pot);

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

        this->vertex->el_prop_action = this->new_action;

        weight::LKMatrix::computeRightSide(cfg->diagram_head, cfg->diagram_tail);
    }
};

#endif // !CHG_TAU_HPP
