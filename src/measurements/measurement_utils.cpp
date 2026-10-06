#include "measurements/measurement_utils.hpp"
#include <cassert>
#include <cmath>
#include <Eigen/Core>
#include "comp_method/weight_computation.hpp"

namespace measurement {

    long double order0Integral(const std::array<double, 3> & p, double chem_pot, double tau_max){
        // order-0 weight: tr e^{-H_LK(p) tau} e^{mu tau} = sum_n e^{-(E_n - mu) tau}, E_n = |p|^2 lambda_n
        // (the masses the diagram uses are 1/(2 lambda_n), and E = k^2/(2m)); the band order is irrelevant here
        const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(p)};
        const long double p_sq {static_cast<long double>(p[0])*p[0] + static_cast<long double>(p[1])*p[1] + static_cast<long double>(p[2])*p[2]};
        long double z0 {0.0L};
        for (int n {0}; n < 3; ++n) {
            const long double rate {p_sq * static_cast<long double>(eig(0, n)) - static_cast<long double>(chem_pot)};
            assert(rate > 0.0L);   // E_n >= 0 and mu < 0
            z0 += -std::expm1(-rate * static_cast<long double>(tau_max)) / rate;
        }
        return z0;
    }

    long double binScale(int bin, double bin_width, double chem_pot){
        return std::exp(-static_cast<long double>(chem_pot) * static_cast<long double>(bin) * static_cast<long double>(bin_width));
    }

    long double binCentre(int bin, double bin_width){
        return (static_cast<long double>(bin) + 0.5L) * static_cast<long double>(bin_width);
    }
}
