#ifndef SEGMENT_ACTION_HPP
#define SEGMENT_ACTION_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <Eigen/Core>

namespace weight {

    // Band-normalised action of an electron segment with band energies E_n and duration d:
    //     action = diag(e^{-(E_n - E_min) d})   (entries in (0, 1], the lowest band exactly 1)
    //     shift  = E_min d
    // so that the full action is e^{-shift} * action. Every cached and staged action is stored this way:
    // a diagram's trace is then T = T~ e^{-S}, S = sum of the segments' shifts, with T~ built from the
    // normalised actions only. T itself underflows once S passes ~745 (long tau_D, energetic segments);
    // T~ does not, and every acceptance ratio is formed as (T~'/T~) e^{-(S' - S)}, where S' - S only
    // involves the segments a move replaces. The off-diagonal entries of `action` are left untouched
    // (they are zero in every action matrix).
    inline void setSegmentAction(Eigen::Matrix3d & action, double & shift, const std::array<double, 3> & energies, double duration){
        const double e_min {std::min({energies[0], energies[1], energies[2]})};
        action(0,0) = std::exp(-(energies[0] - e_min)*duration);
        action(1,1) = std::exp(-(energies[1] - e_min)*duration);
        action(2,2) = std::exp(-(energies[2] - e_min)*duration);
        shift = e_min*duration;
    }

    // the same from k^2 and the band masses, E_n = k^2/(2 m_n)
    inline void setSegmentAction(Eigen::Matrix3d & action, double & shift, double k_sq, const std::array<double, 3> & eff_masses, double duration){
        setSegmentAction(action, shift, std::array<double, 3>{k_sq/(2*eff_masses[0]), k_sq/(2*eff_masses[1]), k_sq/(2*eff_masses[2])}, duration);
    }
}

#endif // !SEGMENT_ACTION_HPP
