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
        inline double compute(
                const std::array<double, 3>& w,
                const double& ph_energy,
                const double& diel_response,
                const double& eff_mass = 1
                ) {
            return ((1./std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]))*std::sqrt(2.*std::sqrt(2.)*std::numbers::pi*std::pow(ph_energy,1.5)*alpha(ph_energy, diel_response, eff_mass)
                    /(Parameters::V_unit_cell*Parameters::V_BvK*std::sqrt(eff_mass))));
            }

        // |g(w)|^2 |w|^2 for a phonon line (both vertices): independent of w, since compute() goes
        // as 1/|w|. This is the line's coupling in the measure dr dOmega, the one the momentum
        // proposal (proposal::PhononMomentum) is written in.
        inline double squaredTimesMomentumSquared(const double& ph_energy, const double& diel_response) {
            const double g_unit {compute(std::array<double, 3>{1., 0., 0.}, ph_energy, diel_response)};
            return g_unit * g_unit;
        }

        }
    
    namespace LKOverlap{
        inline double compute(const Eigen::Vector3d& c1, const Eigen::Vector3d& c2) {
            return c1.dot(c2);
        }

        // Band-basis change between two directions, i.e. compute() above done for all nine
        // (n',n) pairs at once: m1/m2 hold eigenvectors as COLUMNS, so the overlap
        // <n',k'|n,k> = sum_m <n',k'|m><m|n,k> is m1^T * m2, not m1 * m2 (real symmetric
        // H_LK, so the eigenvector matrices are real orthogonal and ^T is the inverse).
        // This is the s(k')s(k)^dagger structure of Guster et al. Eq. (44). Two properties
        // the missing transpose used to break: a vertex that does not rotate the band basis
        // must give the identity (m1^T*m1 == I, while m1*m1 does not), and the trace must be
        // invariant under the arbitrary eigenvector sign gauge fixed in
        // diagonalizeLKHamiltonian - with ^T the signs meet across a diagonal action matrix
        // and cancel, without it they do not.
        inline Eigen::Matrix3d computeMatrix(const Eigen::Matrix3d& m1, const Eigen::Matrix3d& m2){
            return Eigen::Matrix3d(m1.transpose() * m2);
        }
    }
}

#endif // !VERTEX_COUPLING_HPP
