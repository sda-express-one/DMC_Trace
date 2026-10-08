#ifndef STR_DIAGRAM_HPP
#define STR_DIAGRAM_HPP

#include "comp_method/weight_computation.hpp"
#include "diagram/diagram_config.hpp"
#include "utils/sign_counter.hpp"
#include <cassert>
#include <random>
#include <simplemc/random/xoshiro256.hpp>
#include <vector>

// Re-times the whole diagram: the topology, every momentum and the band bases stay, and every segment
// duration D_i (head ... last segment, so tau_D too) is redrawn from an exponential with rate
//     a_i = E_min,i - mu + Omega_i,   Omega_i = summed omega of the phonon lines covering segment i
// (an external line covering a segment twice counts twice). The rates depend only on topology and
// momenta, so the reverse move draws from the same density, and the ratio is the quotient of the
// traces built with band-normalised actions (see attempt()). Proposals beyond tau_max are rejected.
struct str_diagram_update {
    diagram_cfg * const cfg {nullptr};
    simplemc::xoshiro256ss * rng {nullptr};
    mutable std::uniform_real_distribution<double> std_unif {0, 1};
    SignCounter sign;
    std::vector<double> proposed_tau_values;                     // new vertex times, head (0) ... tail (tau_D)

    str_diagram_update(diagram_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);
        
        proposed_tau_values.reserve(cfg->max_vertices);
    }

    str_diagram_update(const str_diagram_update&) = delete;
    str_diagram_update& operator = (const str_diagram_update&) = delete;
    str_diagram_update& operator = (str_diagram_update&&) = delete;
    str_diagram_update(str_diagram_update&& other) noexcept = default;

    double attempt();
    
    void accept();

    void reject(){ sign.rejected(cfg); }
};

#endif // !STR_DIAGRAM_HPP
