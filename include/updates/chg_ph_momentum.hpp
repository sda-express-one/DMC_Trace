#ifndef CHG_PH_MOMENTUM_HPP
#define CHG_PH_MOMENTUM_HPP

#include <array>
#include <cassert>
#include <random>
#include <vector>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/diagram_config.hpp"
#include "comp_method/weight_computation.hpp"

struct chg_ph_momentum {
    diagram_cfg * const cfg {nullptr};
    simplemc::xoshiro256ss* rng {nullptr};
    Vertex * ptr_one {nullptr};
    Vertex * ptr_two {nullptr};
    std::array<double, 3> w_proposed {0., 0., 0.};
    std::vector<weight::ProposedVertexWeight> proposed_weights;

    chg_ph_momentum(diagram_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);

        proposed_weights.reserve(cfg->max_vertices);
    }

    chg_ph_momentum(const chg_ph_momentum&) = delete;
    chg_ph_momentum& operator=(const chg_ph_momentum&) = delete;
    chg_ph_momentum& operator=(chg_ph_momentum&&) = delete;
    chg_ph_momentum(chg_ph_momentum&& other) noexcept = default;

    double attempt();

    void accept(){}
};

#endif // !CHG_PH_MOMENTUM_HPP
