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

        inline double compute(
                const std::array<double, 3>& w,
                const double& ph_energy,
                const double& diel_response,
                const double& eff_mass = 1
                ) {
            return ((1./(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]))*std::sqrt(2.*std::sqrt(2.)*std::numbers::pi*std::pow(ph_energy,1.5)*alpha(ph_energy, diel_response, eff_mass)
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
