#ifndef PHONON_MANAGER_HPP
#define PHONON_MANAGER_HPP

#include <cassert>
#include <random>
#include <simplemc/random/xoshiro256.hpp>
#include <vector>

struct PhononMode {
    const double phonon_energy {0.};
    const double diel_response {1.};
};

struct PhononModeManager{
    std::vector<PhononMode> phonon_mode_pool;
    mutable simplemc::xoshiro256ss * rng {nullptr};
    const int num_phonon_modes {1};

    PhononModeManager(int num_phonon_modes, std::vector<PhononMode> phonon_modes, simplemc::xoshiro256ss * rng)
        : phonon_mode_pool(std::move(phonon_modes)), rng(rng), num_phonon_modes(num_phonon_modes) {
        assert(rng != nullptr);
        assert(static_cast<int>(phonon_mode_pool.size()) == num_phonon_modes);
    }

    const int drawPhononMode() const {
        assert(rng != nullptr);
        assert(num_phonon_modes > 0);

        std::uniform_int_distribution<int> select {0, num_phonon_modes - 1};
        const int position {select(*rng)};

        return position;
    }
};

#endif // !PHONON_MANAGER_HPP
