#ifndef RM_INTERNAL_PH_HPP
#define RM_INTERNAL_PH_HPP

#include <cassert>
#include <vector>
#include <simplemc/random/xoshiro256.hpp>
#include "../diagram/vertex.hpp"
#include "../comp_method/weight_computation.hpp"
#include "../diagram/diagram_config.hpp"


struct rm_int_ph_update {
    diagram_cfg * const cfg {nullptr};

    simplemc::xoshiro256ss * rng {nullptr};
    Vertex * ptr_one {nullptr};
    Vertex * ptr_two {nullptr};

    std::vector<weight::ProposedVertexWeight> proposed_weights;


    rm_int_ph_update(diagram_cfg * cfg, simplemc::xoshiro256ss *rng)
        : cfg(cfg), rng(rng) 
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);

        proposed_weights.reserve(cfg->max_vertices);
    }

    rm_int_ph_update(const rm_int_ph_update&) = delete;
    rm_int_ph_update& operator=(const rm_int_ph_update&) = delete;
    rm_int_ph_update& operator=(rm_int_ph_update&&) = delete;
    rm_int_ph_update(rm_int_ph_update&& other) noexcept = default;
    
    double attempt();
    
    void accept();
};

#endif // !RM_INTERNAL_PH_HPP
