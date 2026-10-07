#ifndef MV_VERTEX_HPP
#define MV_VERTEX_HPP

#include <cassert>
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

    double attempt();

    void accept();

    void reject(){ sign.rejected(cfg); }
};

#endif // !MV_VERTEX_HPP

