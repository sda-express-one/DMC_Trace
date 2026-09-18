#ifndef RM_EXTERNAL_PH_HPP
#define RM_EXTERNAL_PH_HPP

#include <cassert>
#include <vector>
#include <simplemc/random/xoshiro256.hpp>
#include "../diagram/diagram_config.hpp"
#include "../diagram/vertex.hpp"
#include "../comp_method/weight_computation.hpp"

struct rm_ext_ph_update {
    diagram_cfg * const cfg {nullptr};

    simplemc::xoshiro256ss * rng {nullptr};
    Vertex * ptr_one {nullptr};
    Vertex * ptr_two {nullptr};
    int ph_index {-1};

    bool incoming_before_outgoing {true};

    std::vector<weight::ProposedVertexWeight> proposed_weights_beginning;
    std::vector<weight::ProposedVertexWeight> proposed_weights_middle;
    std::vector<weight::ProposedVertexWeight> proposed_weights_end;

    rm_ext_ph_update(diagram_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);

        proposed_weights_beginning.reserve(cfg->max_vertices);
        proposed_weights_end.reserve(cfg->max_vertices);
        proposed_weights_middle.reserve(cfg->max_vertices);
    }

    rm_ext_ph_update(const rm_ext_ph_update&) = delete;
    rm_ext_ph_update& operator=(const rm_ext_ph_update&) = delete;
    rm_ext_ph_update& operator=(rm_ext_ph_update&&) = delete;
    rm_ext_ph_update(rm_ext_ph_update&& other) noexcept = default;

    double attempt();

    void accept();
};

#endif // !RM_EXTERNAL_PH_HPP
