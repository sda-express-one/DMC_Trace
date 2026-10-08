#ifndef VERTEX_HPP
#define VERTEX_HPP

#include <array>
#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include "comp_method/vertex_coupling.hpp"
#include "comp_method/segment_action.hpp"


struct Vertex {
    double tau {0.}; // time coordinate of vertex
    double tau_next {0.}; // time coordinate of following vertex
    int type {0}; // +1 outgoing and -1 incoming (internal), +2 outgoing and -2 incoming (external), 0 unassigned
    int index {-1};
    std::array<double, 3> k {0., 0., 0.}; // momentum components of electron propagator
    std::array<double, 3> w {0., 0., 0,}; // momentum components of phonon propagator
    
    // electronic variables
    std::array<double, 3> eff_masses {1., 1., 1.};
    Eigen::Matrix3d el_prop_action {Eigen::Matrix3d::Identity()};  // band-normalised (weight::setSegmentAction)
    double action_shift {0.};                                        // E_min * duration: full action = e^{-action_shift} el_prop_action
    Eigen::Matrix3d baseWF {Eigen::Matrix3d::Identity()};

    // phononic variables
    double ph_energy {1.};
    double diel_response{1.};
    double ph_action = {0.};

    // el-ph variables
    Eigen::Matrix3d vertex_strength_component {Eigen::Matrix3d::Identity()};
    double vertex_strength_component_scalar {0.};
    Eigen::Matrix3d vertex_wf_component {Eigen::Matrix3d::Identity()};

    Vertex * prev {nullptr}; // link to previous vertex
    Vertex * next {nullptr}; // link to following vertex
    Vertex * conj_vertex {nullptr};

    // matrices for trace weights
    Eigen::Matrix3d right_component {Eigen::Matrix3d::Identity()}; // component to right (to end of diagram)
    Eigen::Matrix3d left_component {Eigen::Matrix3d::Identity()}; // component to the left (to beginning of diagram)
    
    inline std::array<double, 3> electronEnergy() const {
        assert(eff_masses[0] != 0);
        assert(eff_masses[1] != 0);
        assert(eff_masses[2] != 0);
        double k_sq = {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};

        return std::array<double, 3> {k_sq/(2*eff_masses[0]), k_sq/(2*eff_masses[1]), k_sq/(2*eff_masses[2])};
    }

    inline double electronEnergy(int index) const {
        assert(eff_masses[index] != 0);
        double k_sq = {k[0]*k[0] + k[1]*k[1] + k[2]*k[2]};
        return k_sq/(2*eff_masses[index]);
    }

    // the segment's band-normalised action and its shift (see weight::setSegmentAction)
    inline void computeElPropAction(){
        assert(tau_next > tau);
        weight::setSegmentAction(this->el_prop_action, this->action_shift, this->electronEnergy(), tau_next - tau);
    }

    inline double phononEnergy() const {
        return ph_energy;
    }

    inline void computeInternalPhPropAction(){
        assert(conj_vertex != nullptr);
        assert(std::abs(type)%2 != 0);
        assert(this->ph_energy != 0);
        assert(this->ph_energy == conj_vertex->ph_energy);

        this->ph_action = std::exp(-ph_energy*std::abs(this->tau - conj_vertex->tau));
        conj_vertex->ph_action = std::exp(-ph_energy*std::abs(this->tau - conj_vertex->tau));
    }

    inline void computeExternalPhPropAction(double tau_diagram){
        assert(tau_diagram > 0);
        assert(conj_vertex != nullptr);
        assert(type != 0);
        assert(std::abs(type)%2 == 0);
        assert(this->ph_energy != 0);
        assert(this->ph_energy == conj_vertex->ph_energy);

        double tau_interval {0.};
        if(this->type > 0){
           tau_interval = tau_diagram - this->tau + conj_vertex->tau;
        }
        else {
            tau_interval = tau_diagram - conj_vertex->tau + this->tau;
        }
        this->ph_action = std::exp(-ph_energy*tau_interval);
        conj_vertex->ph_action = std::exp(-ph_energy*tau_interval);
    }

    void vertexStrength() {
        vertex_strength_component(0,0) = Coupling::Strength::compute(w, ph_energy, diel_response, eff_masses[0]);
        vertex_strength_component(1,1) = Coupling::Strength::compute(w, ph_energy, diel_response, eff_masses[1]);
        vertex_strength_component(2,2) = Coupling::Strength::compute(w, ph_energy, diel_response, eff_masses[2]);
    }
    
    // in most simulations the effective mass doesn't affect the strength of the interaction
    void vertexStrengthScalar() {
        vertex_strength_component_scalar = Coupling::Strength::compute(w, ph_energy, diel_response);
    }
    
    void vertexOverlap() {
        vertex_wf_component = Coupling::LKOverlap::computeMatrix(prev->baseWF, this->baseWF);
    }

};

#endif // !VERTEX_HPP
