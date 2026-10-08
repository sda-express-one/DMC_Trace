#ifndef SCL_DIAGRAM_HPP
#define SCL_DIAGRAM_HPP

#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include <vector>
#include "diagram/diagram_config.hpp"
#include "utils/sign_counter.hpp"

// Scales the whole diagram: every vertex time and tau_D are multiplied by one factor lambda, so the
// topology, the momenta, the bases and the overlaps stay, and so does the shape u_i = tau_i / tau_D.
// With n vertices the measure is tau_D^n du dtau_D, and along the scaling ray the weight goes as
//     tau_D^n T~(tau_D) e^{-a tau_D},   a = sum_i (E_min,i + Omega_i - mu) D_i / tau_D
// (Omega_i the summed omega of the phonon lines covering segment i, D_i its duration; a depends only on
// the shape, so the move does not change it). tau_D' is drawn from Gamma(n + 1, a), which is that
// scalar part exactly and does not depend on the current tau_D: the Jacobian, mu, the phonon
// propagators and the lowest-band exponentials all cancel, and the ratio is T~(tau_D')/T~(tau_D), the
// band-normalised traces (the current one is the cache). Proposals beyond tau_max are rejected.
// Changes tau_D: not to be used in fixed-length runs.
struct scl_diagram_update {
    diagram_cfg * const cfg {nullptr};
    simplemc::xoshiro256ss * rng {nullptr};
    SignCounter sign;
    std::vector<double> proposed_tau_values;   // new vertex times, head (0) ... tail (tau_D')

    scl_diagram_update(diagram_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);

        proposed_tau_values.reserve(cfg->max_vertices);
    }

    scl_diagram_update(const scl_diagram_update&) = delete;
    scl_diagram_update& operator=(const scl_diagram_update&) = delete;
    scl_diagram_update& operator=(scl_diagram_update&&) = delete;
    scl_diagram_update(scl_diagram_update&& other) noexcept = default;

    double attempt();

    void accept();

    void reject(){ sign.rejected(cfg); }
};

#endif // !SCL_DIAGRAM_HPP
