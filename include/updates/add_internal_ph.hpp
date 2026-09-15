#ifndef ADD_INTERNAL_PH_HPP
#define ADD_INTERNAL_PH_HPP

#include <array>
#include <cassert>
#include <random>
#include <vector>
#include <simplemc/random/xoshiro256.hpp>
#include "../vertex.hpp"
#include "../weight_computation.hpp"
#include "updates_config.hpp"


struct add_int_ph_update {
    updates_cfg * const cfg {nullptr}; // bound to one diagram/config for this object's whole
                                        // lifetime - never reseated, so proposed_weights' reserved
                                        // capacity (sized from cfg->max_vertices below) can't go stale
    simplemc::xoshiro256ss * rng {nullptr};
    mutable std::uniform_real_distribution<double> std_unif {0,1};
    Vertex* ptr_one {nullptr};
    Vertex* ptr_two {nullptr};
    double tau_one {0.};
    double tau_two {0.};
    int ph_index {-1};
    std::array<double, 3> w_proposed {0., 0., 0.};
    std::vector<weight::ProposedVertexWeight> proposed_weights;

    add_int_ph_update(updates_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);

        // reserved once up front to the largest the diagram can ever grow, so
        // filling/clearing it while walking ptr_one..ptr_two during attempt()
        // never triggers a reallocation.
        proposed_weights.reserve(cfg->max_vertices);
    }

    // proposed_weights' reservation is only actually cheap for the whole run if this object is
    // never copied (a copy of a std::vector only guarantees capacity for its current size, not
    // whatever was reserved) - so copying is disabled outright. Move-construction is kept since
    // it's exactly how simplemc::update<add_int_ph_update> takes ownership of this at setup time,
    // and moving a vector preserves its reserved capacity intact.
    add_int_ph_update(const add_int_ph_update&) = delete;
    add_int_ph_update& operator=(const add_int_ph_update&) = delete;
    add_int_ph_update& operator=(add_int_ph_update&&) = delete;
    add_int_ph_update(add_int_ph_update&& other) noexcept = default;

    double attempt();

    void accept();
};

#endif // !ADD_INTERNAL_PH_HPP
