#ifndef MV_VERTEX_HPP
#define MV_VERTEX_HPP

#include <cassert>
#include <cmath>
#include <random>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/vertex.hpp"
#include "diagram/diagram_config.hpp"
#include "utils/sign_counter.hpp"
#include "comp_method/weight_computation.hpp"

struct mv_tau_update {
    diagram_cfg * const cfg;
    simplemc::xoshiro256ss* rng;
    SignCounter sign;   // sign of the ratio, negative-diagram count (utils/sign_counter.hpp)
    mutable std::uniform_real_distribution<double> std_unif {0.,1.};
    Vertex * vertex {nullptr};
    Eigen::Matrix3d new_action_el_incoming {Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d new_action_el_outgoing {Eigen::Matrix3d::Identity()};
    int index {-1};
    double tau_proposed {0.};

    mv_tau_update(diagram_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);
    }

    mv_tau_update(const mv_tau_update&) = delete;
    mv_tau_update& operator=(const mv_tau_update&) = delete;
    mv_tau_update& operator=(mv_tau_update&&) = delete;
    mv_tau_update(mv_tau_update&& other) noexcept = default;

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

        const double deltaE_electronic {lowest_energy_incoming - lowest_energy_outgoing};
        const double deltaE {deltaE_electronic - ph_energy};

        // draw tau from p(tau) ~ exp(-deltaE*(tau - tau_prev)) truncated to [tau_prev, tau_next], i.e.
        // tau = tau_prev - log(1 - u(1 - exp(-a)))/deltaE with a = deltaE*L. Evaluated that way it
        // overflows for a < -709.8 (exp(-a) = inf -> tau = +inf), which happens whenever the outgoing
        // segment is far more energetic than the incoming one. expm1/log1p keep every intermediate in
        // [-1, 0], and for a < 0 the distance is measured from tau_next instead - the mirror image of
        // the same truncated exponential - so the draw stays exact near the endpoint it piles up at.
        const double L {tau_next - tau_prev};
        const double a {deltaE * L};
        const double u {std_unif(*rng)};
        if (a > 0.) {
            tau_proposed = tau_prev - L * std::log1p(u * std::expm1(-a)) / a;
        }
        else if (a < 0.) {
            tau_proposed = tau_next - L * std::log1p(u * std::expm1(a)) / a;
        }
        else {
            tau_proposed = tau_prev + u * L;
        }

        // the endpoints have probability zero but can be hit in floating point (u = 0, or rounding
        // onto tau_next); a vertex there would leave a zero-length segment, so reject the draw. The
        // condition is also false for a non-finite tau_proposed.
        if (!(tau_proposed > tau_prev && tau_proposed < tau_next)) {
            return -1.;
        }

        // full actions of the two re-timed segments, committed by accept()
        new_action_el_incoming(0,0) = std::exp(-energies_incoming[0]*(tau_proposed-tau_prev));
        new_action_el_incoming(1,1) = std::exp(-energies_incoming[1]*(tau_proposed-tau_prev));
        new_action_el_incoming(2,2) = std::exp(-energies_incoming[2]*(tau_proposed-tau_prev));

        new_action_el_outgoing(0,0) = std::exp(-energies_outgoing[0]*(tau_next-tau_proposed));
        new_action_el_outgoing(1,1) = std::exp(-energies_outgoing[1]*(tau_next-tau_proposed));
        new_action_el_outgoing(2,2) = std::exp(-energies_outgoing[2]*(tau_next-tau_proposed));

        // Acceptance ratio. Detailed balance needs W(y)/W(x) * p(tau_x)/p(tau_y) with the proposal
        // density p above; its normalisation cancels (deltaE and L do not depend on tau_v, so the reverse
        // move draws from the same p), and its phonon part (-ph_energy) cancels exactly against the
        // tau_v-dependence of the phonon propagator, which is therefore left out of W as well. What
        // remains, exp(-deltaE_electronic*(tau_x - tau_y)) * W_el(y)/W_el(x), is evaluated without
        // forming it: factoring each segment's action as exp(-E_min*d) * diag(exp(-(E_i - E_min)*d)),
        // the lowest-band exponentials of the two segments reproduce exactly that prefactor and cancel
        // it, so the ratio is the quotient of the same trace built with the band-normalised actions
        // only. Every normalised entry lies in (0, 1], and no absolute time or large exponent appears -
        // evaluating exp(-deltaE_electronic*tau) on its own overflows once |deltaE_electronic|*tau > 709.
        auto band_normalised = [](const std::array<double, 3>& energies, double e_min, double duration){
            return Eigen::Matrix3d(Eigen::Vector3d(
                std::exp(-(energies[0] - e_min)*duration),
                std::exp(-(energies[1] - e_min)*duration),
                std::exp(-(energies[2] - e_min)*duration)).asDiagonal());
        };

        // both diagrams share everything outside [tau_prev, tau_next]: L(prev) and R(next) are reused
        auto trace_with = [this](const Eigen::Matrix3d& action_in, const Eigen::Matrix3d& action_out){
            return (
                vertex->prev->vertex_wf_component *
                action_in *
                vertex->vertex_wf_component *
                action_out *
                vertex->next->vertex_wf_component *
                vertex->next->right_component *
                vertex->prev->left_component
            ).trace();
        };

        const double diagram_weight_current {trace_with(
            band_normalised(energies_incoming, lowest_energy_incoming, vertex->tau - tau_prev),
            band_normalised(energies_outgoing, lowest_energy_outgoing, tau_next - vertex->tau))};
        const double diagram_weight_proposed {trace_with(
            band_normalised(energies_incoming, lowest_energy_incoming, tau_proposed - tau_prev),
            band_normalised(energies_outgoing, lowest_energy_outgoing, tau_next - tau_proposed))};

        return sign.take(diagram_weight_proposed / diagram_weight_current);
    }

    void accept(){
        this->vertex->prev->tau_next = tau_proposed;
        this->vertex->tau = tau_proposed;
        
        this->vertex->prev->el_prop_action = new_action_el_incoming;
        this->vertex->el_prop_action =  new_action_el_outgoing;

        weight::LKMatrix::computeRightSide(this->cfg->diagram_head, this->vertex->next);
        weight::LKMatrix::computeLeftSide(this->cfg->diagram_tail, this->vertex->prev);

        std::abs(this->vertex->type) % 2 == 1 ? this->vertex->computeInternalPhPropAction() : this->vertex->computeExternalPhPropAction(this->cfg->diagram_tail->tau);

        sign.accepted(cfg);
    }

    void reject(){
        sign.rejected(cfg);
    }
};

#endif // !MV_VERTEX_HPP

