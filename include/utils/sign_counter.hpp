#ifndef SIGN_COUNTER_HPP
#define SIGN_COUNTER_HPP

#include <cmath>
#include <cstdint>
#include "diagram/diagram_config.hpp"

// Sign bookkeeping shared by the updates. The chain samples |W|: attempt() passes its (signed) ratio
// through take(), which remembers the sign and returns the absolute value to the Metropolis kernel.
// The kernel then calls exactly one of accept() / reject() per step (simplemc's metropolis_kernel,
// impossible proposals included), which call accepted() / rejected() here: the diagram's sign is
// updated as current_sign *= sign(ratio) on accept and left unchanged on reject, and n_negative counts
// this update's steps that end on a negative diagram (positives = simplemc's nprops - n_negative).
// The -1 "impossible" sentinels of attempt() must not go through take().
struct SignCounter {
    int proposed_sign {1};          // sign of the last genuine ratio
    std::uint64_t n_negative {0};   // steps of this update ending on a negative diagram

    double take(double ratio) {
        proposed_sign = ratio < 0. ? -1 : 1;
        return std::abs(ratio);
    }

    void accepted(diagram_cfg * cfg) {
        cfg->current_sign *= proposed_sign;
        if (cfg->current_sign < 0) { ++n_negative; }
    }

    void rejected(const diagram_cfg * cfg) {
        if (cfg->current_sign < 0) { ++n_negative; }
    }
};

#endif // !SIGN_COUNTER_HPP
