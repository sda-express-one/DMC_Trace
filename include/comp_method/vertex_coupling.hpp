#ifndef VERTEX_COUPLING_HPP
#define VERTEX_COUPLING_HPP

#include <array>
#include <cmath>
#include <numbers>
#include <Eigen/Core>

namespace Coupling {
    struct Parameters{
        inline static double V_unit_cell {1.};
        inline static double V_BvK {1.};

        void inline static configParameters(double V_unit_cell, double V_BvK){
            Coupling::Parameters::V_unit_cell = V_unit_cell;
            Coupling::Parameters::V_BvK = V_BvK;
        }
    };

    namespace Strength {
        inline double alpha(const double& ph_energy, const double& diel_response, const double& eff_mass = 1) {
            return ((1./diel_response)*std::sqrt(eff_mass/(2*ph_energy)));
        }

        // |g(q)| for ONE vertex, so a phonon line - which has two - contributes this squared.
        // Leading factor is 1/q, not 1/q^2: squaring gives the 1/q^2 of |g|^2, matching the
        // i/q prefactor of Guster et al. Eq. (5)/(6). The eff_mass dependence cancels
        // identically (alpha goes as sqrt(m), divided by sqrt(m) here), as it must - Eq. (6)
        // with Eq. (7) substituted reduces to Eq. (5), which carries no m*.
        // V_BvK is absent on purpose: the caller's acceptance ratio divides by (2*pi)^3 with
        // no compensating box volume, i.e. it works in the continuum limit where the discrete
        // sum has already been converted as sum_q -> V_BvK/(2*pi)^3 * integral d^3q. Keeping
        // V_BvK here would leave a stray 1/V_BvK in every ratio. V_unit_cell, by contrast,
        // stays: every call site multiplies its ratio by V_unit_cell on the same side as this
        // pair, so the 1/sqrt(V_unit_cell) here squares away against it exactly.
        inline double compute(
                const std::array<double, 3>& w,
                const double& ph_energy,
                const double& diel_response,
                const double& eff_mass = 1
                ) {
            return ((1./std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]))*std::sqrt(2.*std::sqrt(2.)*std::numbers::pi*std::pow(ph_energy,1.5)*alpha(ph_energy, diel_response, eff_mass)
                    /(Parameters::V_unit_cell*Parameters::V_BvK*std::sqrt(eff_mass))));
            }
                
        }
    
    namespace LKOverlap{
        inline double compute(const Eigen::Vector3d& c1, const Eigen::Vector3d& c2) {
            return c1.dot(c2);
        }

        inline Eigen::Matrix3d computeMatrix(const Eigen::Matrix3d& m1, const Eigen::Matrix3d& m2){
            return Eigen::Matrix3d(m1 * m2);
        }
    }
}

#endif // !VERTEX_COUPLING_HPP
