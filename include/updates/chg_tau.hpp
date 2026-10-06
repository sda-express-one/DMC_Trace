#ifndef CHG_TAU_HPP
#define CHG_TAU_HPP

#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/vertex.hpp"
#include "comp_method/weight_computation.hpp"
#include "diagram/diagram_config.hpp"
#include "utils/sign_counter.hpp"

struct chg_tau_update {
    diagram_cfg * const cfg;
    simplemc::xoshiro256ss* rng;
    SignCounter sign;   // sign of the ratio, negative-diagram count (utils/sign_counter.hpp)
    mutable std::uniform_real_distribution<double> std_unif {0.,1.};
    Vertex * vertex {nullptr};
    Eigen::Matrix3d new_action {Eigen::Matrix3d::Identity()};
    double tau_last_vertex {0.};
    double tau_proposed {0.};

    chg_tau_update(diagram_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);
    }

    chg_tau_update(const chg_tau_update&) = delete;
    chg_tau_update& operator=(const chg_tau_update&) = delete;
    chg_tau_update& operator=(chg_tau_update&&) = delete;
    chg_tau_update(chg_tau_update&& other) noexcept = default;

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

        // u = 0 gives tau_proposed == tau_last_vertex exactly: a zero-length last segment. It has
        // probability zero but can be drawn in floating point, so reject it. Written as a negated
        // comparison so it also rejects a NaN tau_proposed.
        if(!(tau_proposed > this->tau_last_vertex)){
            return -1.;
        }

        this->new_action(0,0) = std::exp(-energies[0]*(tau_proposed - this->tau_last_vertex));
        this->new_action(1,1) = std::exp(-energies[1]*(tau_proposed - this->tau_last_vertex));
        this->new_action(2,2) = std::exp(-energies[2]*(tau_proposed - this->tau_last_vertex));

        // Acceptance ratio. The proposal rate total_lowest_energy = E_min - mu + sum(omega_ext) does not
        // depend on the current tail position, so the reverse move draws from the same density; its mu
        // and external-phonon parts cancel exactly against the tau_D-dependence of exp(mu*tau_D) and of
        // the external phonon propagators, which are therefore left out of the weight as well. What
        // remains is exp(-E_min*(tau_D - tau_proposed)) * T(proposed)/T(current), with T the electronic
        // trace. It is evaluated without forming that factor: writing the last segment's action as
        // exp(-E_min*d) * diag(exp(-(E_i - E_min)*d)), the lowest-band exponentials of the two
        // durations reproduce exactly that prefactor and cancel it, so the ratio is the quotient of the
        // same trace built with the band-normalised action only. Every normalised entry lies in (0, 1]
        // and the absolute time tau_D never appears - evaluating exp(-E_min*tau_D) on its own underflows
        // once E_min*tau_D exceeds ~708 (subnormal, losing digits) and is 0 beyond ~745, which gave 0/0
        // for every attempt in long diagrams with an energetic last segment.
        auto band_normalised = [&energies, lowest_energy](double duration){
            return Eigen::Matrix3d(Eigen::Vector3d(
                std::exp(-(energies[0] - lowest_energy)*duration),
                std::exp(-(energies[1] - lowest_energy)*duration),
                std::exp(-(energies[2] - lowest_energy)*duration)).asDiagonal());
        };

        // everything before the last segment is shared: L(vertex) and vertex's own overlap are reused
        auto trace_with = [this](const Eigen::Matrix3d& action){
            return (vertex->vertex_wf_component * action * vertex->left_component).trace();
        };

        const double diagram_weight_current {trace_with(band_normalised(this->cfg->diagram_tail->tau - this->tau_last_vertex))};
        const double diagram_weight_proposed {trace_with(band_normalised(tau_proposed - this->tau_last_vertex))};

        return sign.take(diagram_weight_proposed / diagram_weight_current);
    }

    void accept(){
        this->vertex->tau_next = tau_proposed;
        this->cfg->diagram_tail->tau = tau_proposed;
        this->cfg->current_tau_length = tau_proposed;

        this->vertex->el_prop_action = this->new_action;

        weight::LKMatrix::computeRightSide(cfg->diagram_head, cfg->diagram_tail);
        // only the last segment's action changed: every left_component up to the last vertex is
        // unaffected (none includes that action), but the tail's - the full product through the last
        // segment - does. Refreshing from the last vertex reseeds from its own left_component, so this
        // is O(1): it sets tail->left_component = L(vertex) * wf(vertex) * A(vertex).
        weight::LKMatrix::computeLeftSide(cfg->diagram_tail, this->vertex);

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

        sign.accepted(cfg);
    }

    void reject(){
        sign.rejected(cfg);
    }
};

#endif // !CHG_TAU_HPP
