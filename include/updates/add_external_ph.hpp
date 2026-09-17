#ifndef ADD_EXTERNAL_PH_HPP
#define ADD_EXTERNAL_PH_HPP


#include <array>
#include <cassert>
#include <random>
#include <vector>
#include <simplemc/random/xoshiro256.hpp>
#include "updates_config.hpp"
#include "../vertex.hpp"
#include "../weight_computation.hpp"

struct add_ext_ph_update {
    updates_cfg * const cfg {nullptr};

    simplemc::xoshiro256ss * rng {nullptr};
    mutable std::uniform_real_distribution<double> std_unif {0.,1.};
    Vertex * ptr_one {nullptr};
    Vertex * ptr_two {nullptr};
    double tau_one {0.};
    double tau_two {0.};
    int ph_index {-1};
    std::array<double, 3> w_proposed {0., 0., 0.};

    // set in attempt(), read back in accept(): true for case 1 (incoming/-2 before
    // outgoing/+2, i.e. tau_one < tau_two - only beginning/end change), false for case 2
    // (outgoing before incoming - beginning/end/middle all change, middle double-shifted).
    bool incoming_before_outgoing {true};

    // Three independent linked-list walks, none of which chain into another via ->next:
    // - beginning: diagram_head -> ptr_one (used by both cases)
    // - end:       ptr_two -> diagram_tail (used by both cases)
    // - middle:    ptr_one -> ptr_two directly, i.e. NOT through the wrap (case 2 only,
    //              where creation comes before annihilation)
    std::vector<weight::ProposedVertexWeight> proposed_weights_beginning;
    std::vector<weight::ProposedVertexWeight> proposed_weights_end;
    std::vector<weight::ProposedVertexWeight> proposed_weights_middle;

    add_ext_ph_update(updates_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);

        // each walk is bounded by the same worst case (the whole diagram), so each vector
        // is reserved to the full cfg->max_vertices independently rather than trying to
        // divide that bound between them - see the earlier max_order_int discussion for why
        // a tighter, "smarter" bound is the wrong trade here.
        proposed_weights_beginning.reserve(cfg->max_vertices);
        proposed_weights_end.reserve(cfg->max_vertices);
        proposed_weights_middle.reserve(cfg->max_vertices);
    }

    add_ext_ph_update(const add_ext_ph_update&) = delete;
    add_ext_ph_update& operator=(const add_ext_ph_update&) = delete;
    add_ext_ph_update& operator=(add_ext_ph_update&&) = delete;
    add_ext_ph_update(add_ext_ph_update&& other)  noexcept = default;

    double attempt();

    void accept();
};

#endif // !ADD_EXTERNAL_PH_HPP
