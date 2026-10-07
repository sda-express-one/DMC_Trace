#ifndef CHG_PH_ENERGY_HPP
#define CHG_PH_ENERGY_HPP


#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include "diagram/diagram_config.hpp"
#include "diagram/vertex.hpp"
#include "utils/sign_counter.hpp"

// Changes the phonon mode of one line: a line chosen uniformly (one per even slot of the two managers)
// and a mode drawn uniformly from the PhononModeManager; drawing the line's current mode is rejected
// as impossible. Both choices are the same in the reverse move (the number of lines and of modes does
// not change), so the ratio is the weight ratio alone. Only the line's own factors depend on the mode,
//     r = C(w', e') e^{-w' l} / ( C(w, e) e^{-w l} ),   C = |g|^2|q|^2 (squaredTimesMomentumSquared),
// with l the line length (wrapped for an external line); the momentum, and so 1/|q|^2, is unchanged,
// and the electronic trace does not depend on the mode.
struct chg_ph_energy {
    diagram_cfg * const cfg {nullptr};
    simplemc::xoshiro256ss * rng {nullptr};
    SignCounter sign;
    Vertex * ptr_vertex {nullptr};  // creation vertex of the chosen line (even slot of its pair)
    bool internal {true};           // whether that line is internal (else external, wrapped)
    int ph_index {-1};              // proposed phonon mode

    chg_ph_energy(diagram_cfg * cfg, simplemc::xoshiro256ss * rng)
        : cfg(cfg), rng(rng)
    {
        assert(cfg != nullptr);
        assert(rng != nullptr);
    }

    chg_ph_energy(const chg_ph_energy&) = delete;
    chg_ph_energy& operator=(const chg_ph_energy&) = delete;
    chg_ph_energy& operator=(chg_ph_energy&&) = delete;
    chg_ph_energy(chg_ph_energy&& other) noexcept = default;

    double attempt();

    void accept();

    void reject(){sign.rejected(cfg);}
};

#endif // !CHG_PH_ENERGY_HPP
