#ifndef CHG_TAU_HPP
#define CHG_TAU_HPP

#include <cmath>
#include <random>
#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include "../vertex.hpp"

struct chg_tau_cfg {
    Vertex * diagram_tail {nullptr};
    Vertex * vertex {nullptr}; // pointer to vertex beforen tail
    double tau_current {0.};
    const double tau_max {50.};
    const double chem_pot {1.};

    chg_tau_cfg(Vertex * diagram_tail, double tau_max = 50.0, double chem_pot = 1.0) 
        : diagram_tail(diagram_tail), tau_max(tau_max), chem_pot(chem_pot) {
        assert(this->diagram_tail != nullptr);
    }
};

struct chg_tau_update {
    chg_tau_cfg* cfg;
    simplemc::xoshiro256ss* rng;
    mutable std::uniform_real_distribution<double> std_unif {0.,1.};

    double tau_proposed {0.};

    double attempt(){
        this->cfg->vertex = this->cfg->diagram_tail->prev;
        this->cfg->tau_current = this->cfg->vertex->tau;
        
        // placeholder to change
        tau_proposed = this->cfg->tau_current - std::log(1 - std_unif(*rng))/((this->cfg->vertex->electronEnergy())[0] - this->cfg->chem_pot);

        return -1.;
    }

    void accept(){
        cfg->vertex->tau_next = tau_proposed;
        cfg->diagram_tail->tau = tau_proposed;
    }
};

#endif // !CHG_TAU_HPP
