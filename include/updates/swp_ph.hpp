#ifndef SWP_PH_HPP
#define SWP_PH_HPP

#include <array>
#include <cassert>
#include <random>
#include <simplemc/random/xoshiro256.hpp>
#include "updates_config.hpp"
#include "../vertex.hpp"
#include "../weight_computation.hpp"


struct swp_ph_update {
    updates_cfg * const cfg {nullptr};

    simplemc::xoshiro256ss * rng {nullptr};
    mutable std::uniform_real_distribution<double> std_unif {0.,1.};
 
    Vertex * ptr_one {nullptr};
    Vertex * ptr_two {nullptr};
    std::array<double, 3> k_new {0., 0., 0.};
    std::array<weight::ProposedVertexWeight, 2> proposed_weights;



    swp_ph_update(updates_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);

    }

    swp_ph_update(const swp_ph_update&) = delete;
    swp_ph_update& operator=(const swp_ph_update&) = delete;
    swp_ph_update& operator=(swp_ph_update&&) = delete;
    swp_ph_update(swp_ph_update&& other) noexcept = default;

    double attempt();

    void accept();
};

#endif // !SWP_PH_HPP
