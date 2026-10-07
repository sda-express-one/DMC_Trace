#ifndef CHG_TAU_HPP
#define CHG_TAU_HPP

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

    double attempt();

    void accept();

    void reject(){ sign.rejected(cfg); }
};

#endif // !CHG_TAU_HPP
