#ifndef RM_EXTERNAL_PH_HPP
#define RM_EXTERNAL_PH_HPP

#include <cassert>
#include <vector>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/diagram_config.hpp"
#include "utils/sign_counter.hpp"
#include "diagram/vertex.hpp"
#include "comp_method/weight_computation.hpp"
#include "utils/prop_distribs.hpp"

struct rm_ext_ph_update {
    diagram_cfg * const cfg {nullptr};

    simplemc::xoshiro256ss * rng {nullptr};
    SignCounter sign;   // sign of the ratio, negative-diagram count (utils/sign_counter.hpp)
    Vertex * ptr_one {nullptr};
    Vertex * ptr_two {nullptr};
    // the proposal add_ext_ph draws w from, for the reverse density
    proposal::PhononMomentum w_proposal {};
    int ph_index {-1};

    bool incoming_before_outgoing {true};

    // removing the last external line (case 1): the merge entry before the untouched middle became canonical
    // (k = p, basis U(p)), so the first middle vertex's overlap is restaged in attempt() and committed in accept()
    bool restage_first_middle {false};
    Eigen::Matrix3d first_middle_overlap {Eigen::Matrix3d::Identity()};

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

    void reject(){ sign.rejected(cfg); }
};

#endif // !RM_EXTERNAL_PH_HPP
