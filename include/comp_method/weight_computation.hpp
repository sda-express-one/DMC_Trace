#ifndef WEIGHT_COMPUTATION_HPP
#define WEIGHT_COMPUTATION_HPP

#include <algorithm>
#include <array>
#include <Eigen/Core>
#include <Eigen/src/Core/util/XprHelper.h>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <cassert>
#include <limits>
#include "diagram/vertex.hpp"
#include "comp_method/segment_action.hpp"
#include "utils/numerical.hpp"

namespace weight {
    struct ProposedVertexWeight {
        std::array<double, 3> k {0., 0., 0.};
        std::array<double, 3> eff_masses {1., 1., 1.};
        double electron_energy {0.};
        Eigen::Matrix3d baseWF {Eigen::Matrix3d::Identity()};
        Eigen::Matrix3d el_prop_action {Eigen::Matrix3d::Identity()};  // band-normalised, as on Vertex
        double action_shift {0.};                                        // E_min * duration of the staged segment
        Eigen::Matrix3d vertex_wf_component {Eigen::Matrix3d::Identity()};
    };

    namespace phononic {

    }

    struct LKMatrix {
        struct choice{
            std::array<int, 3> position {0, 1, 2};
            std::array<int, 3> sign {1, 1, 1};
        };
        
        // A_LK > 0
        // B_LK > 0
        // C < A + B
        // C > -1/2 A - B
        inline static double A_LK {1.};
        inline static double B_LK {1.};
        inline static double C_LK {0.};

        static void setLKParameters(double A_LK, double B_LK, double C_LK){
            weight::LKMatrix::A_LK = A_LK;
            weight::LKMatrix::B_LK = B_LK;
            weight::LKMatrix::C_LK = C_LK;
        }

        static choice selectionRulesLK(const std::array<double, 3> k){
            choice eigenv_gauge; // position starts at the identity {0, 1, 2}

            // sort axis indices by |k| descending; stable_sort keeps the lower original
            // index first on an exact tie, matching the previous >= branch cascade
            std::stable_sort(eigenv_gauge.position.begin(), eigenv_gauge.position.end(),
                              [&k](int a, int b){ return std::abs(k[a]) > std::abs(k[b]); });

            for(int i = 0; i < 3; ++i){
                if(k[eigenv_gauge.position[i]] < 0){ eigenv_gauge.sign[i] = -1; }
            }

            return eigenv_gauge;
        }

        // diagonalizes LK Hamiltonian and returns eigenvalues and eigenvectors in correct order
        static Eigen::Matrix<double, 4, 3> diagonalizeLKHamiltonian(const std::array<double, 3> k){
    
            Eigen::Matrix<double, 4, 3> result;
    
            // detects free propagators. At the band extremum H_LK(k) vanishes identically: the
            // three bands are exactly degenerate there and no direction exists (the matrix below
            // is built from the normalised vector, so it would be 0/0). Nothing may therefore be
            // split here - returning a direction-dependent spectrum such as {A, B, B} would pick
            // a fictitious k_hat and lift a degeneracy that is exact.
            // The eigenvalue this function returns is 1/(2m), not an energy, since the matrix is
            // built from k_hat - so the vanishing of H_LK(0) reads as {0, 0, 0} here, and 1 is
            // substituted only because the caller inverts it (computeEffMassfromEigenval), where
            // a zero would give an infinite mass. The substitution costs nothing: every consumer
            // of eff_masses forms k^2/(2m) with this same k, so at k = 0 the mass cancels out of
            // every result (the electron energy is 0 and el_prop_action is 1 whatever it holds),
            // and Strength::compute's m dependence cancels identically. {1, 1, 1} keeps the three
            // bands degenerate and the masses finite and positive.
            // Eigenvectors are the identity, the fixed Gamma basis itself: orthonormal, as every
            // overlap in the trace product assumes, and in the positive-diagonal gauge fixed
            // below. With the eigenvalues degenerate the choice within the eigenspace is free
            // anyway - the propagator is proportional to the identity, so any rotation commutes.
            if(numerical::isEqual(k[0], 0.) && numerical::isEqual(k[1],0.) && numerical::isEqual(k[2],0.)){
                result << 1., 1., 1.,
                          1., 0., 0.,
                          0., 1., 0.,
                          0., 0., 1.;
                return result;
            }

            double k_modulus {std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])};
            double k_vector[3] {k[0]/k_modulus, k[1]/k_modulus, k[2]/k_modulus};
            double k_vector_temp[3] {k[0]/k_modulus, k[1]/k_modulus, k[2]/k_modulus};

            choice eigenv_gauge = selectionRulesLK(k); // cast k-values into IBZ

            k_vector[0] = std::abs(k_vector_temp[eigenv_gauge.position[0]]);
            k_vector[1] = std::abs(k_vector_temp[eigenv_gauge.position[1]]);
            k_vector[2] = std::abs(k_vector_temp[eigenv_gauge.position[2]]);

            // build LK Hamiltonian matrix (3x3)
            Eigen::Matrix3d LK_matrix;
            LK_matrix << A_LK*k_vector[0]*k_vector[0] + B_LK*(k_vector[1]*k_vector[1] + k_vector[2]*k_vector[2]), C_LK*k_vector[0]*k_vector[1], C_LK*k_vector[0]*k_vector[2],
                     C_LK*k_vector[0]*k_vector[1], A_LK*k_vector[1]*k_vector[1] + B_LK*(k_vector[0]*k_vector[0] + k_vector[2]*k_vector[2]), C_LK*k_vector[1]*k_vector[2],
                     C_LK*k_vector[0]*k_vector[2], C_LK*k_vector[1]*k_vector[2], A_LK*k_vector[2]*k_vector[2] + B_LK*(k_vector[0]*k_vector[0] + k_vector[1]*k_vector[1]);
    
            // Bands are always returned in ascending eigenvalue order: band 0 has the smallest 1/(2m), i.e. it
            // is the heaviest band and the lowest in energy, whatever A, B and C are.
            Eigen::Matrix3d eigenvectors;
            Eigen::RowVector3d eigenvalues;

            if(C_LK == 0.){
                // C = 0: the matrix is diagonal, d_i = B + (A - B) k_i^2, so the eigenvectors are the axes
                // themselves - taken exactly instead of from the solver. For A = B the matrix is A * I only
                // up to rounding, and the solver would return a basis set by that noise, different for two
                // momenta one ulp apart (e.g. the first and last segment): the trace would then not close.
                // In the wedge k_0 >= k_1 >= k_2, so the diagonal is descending for A > B and ascending for
                // A < B: for A > B the axes are taken in reverse. The order is fixed by A and B, never by
                // comparing the d_i, which for A = B are equal up to rounding.
                const Eigen::Vector3d d {LK_matrix.diagonal()};
                if(A_LK > B_LK){
                    eigenvectors << 0., 0., 1.,
                                    0., 1., 0.,
                                    1., 0., 0.;
                    eigenvalues << d(2), d(1), d(0);
                }
                else {
                    eigenvectors.setIdentity();
                    eigenvalues = d.transpose();
                }
            }
            else {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
                eigensolver.compute(LK_matrix); // compute eigenvalues and eigenvectors

                if(eigensolver.info() != 0){
                    const double nan {std::numeric_limits<double>::quiet_NaN()};
                    result << nan, nan, nan,
                              nan, nan, nan,
                              nan, nan, nan,
                              nan, nan, nan;
                    return result; // solver failure sentinel (NaN, never a valid eigenvalue)
                }

                eigenvectors = eigensolver.eigenvectors(); // eigenvector matrix
                eigenvalues = eigensolver.eigenvalues().transpose(); // eigenvalues row vector (from smallest to largest)
            }

            // sign gauge: each eigenvector's component along its reference axis is made positive. In the wedge
            // the bands, in ascending order, lie mostly along axes 0, 1, 2 for A <= B and along 2, 1, 0 for
            // A > B (near (100): eigenvalues A on axis 0 and B on axes 1, 2), so the reference axis of band n
            // is n or 2 - n (the same vectors as the earlier positive-diagonal rule applied before reordering;
            // tests/test_lk_sphere checks that the basis is continuous inside every wedge)
            const bool reversed {A_LK > B_LK};
            for (int n {0}; n < 3; ++n) {
                if(eigenvectors(reversed ? 2 - n : n, n) < 0){ eigenvectors.col(n) *= -1; }
            }

            // back from the IBZ to the original sector: the transformation is a signed permutation T with
            // T(position[i], i) = sign[i] and zeros elsewhere, so T * eigenvectors just moves row i to row
            // position[i] with that sign - applied directly instead of as a matrix product
            Eigen::Matrix3d new_eigenvectors;
            for (int i {0}; i < 3; ++i) {
                new_eigenvectors.row(eigenv_gauge.position[i]) = static_cast<double>(eigenv_gauge.sign[i]) * eigenvectors.row(i);
            }

            eigenvectors = new_eigenvectors;

            result << eigenvalues, eigenvectors;

            // return packed result
            return result;
        }

        // diagonalizes LK Hamiltonian and returns eigenvalues (no eigenvectors), for effective mass exact estimator
        static Eigen::RowVector3d diagonalizeLKHamiltonianEigenval(const std::array<double, 3> k){

            Eigen::RowVector3d eigenvalues;
        
            // free propagator: same degenerate eigenvalues as diagonalizeLKHamiltonian's k=(0,0,0)
            // case - H_LK vanishes at the extremum, so nothing is split, and 1 rather than 0 only
            // because the caller inverts it for the mass. See there for the full reasoning.
            // Away from k = 0 this function returns the solver's ascending order, the same band order
            // as diagonalizeLKHamiltonian.
            if(numerical::isEqual(k[0],0) && numerical::isEqual(k[1],0) && numerical::isEqual(k[2],0)){
                eigenvalues << 1., 1., 1.;
                return eigenvalues;
            }
    
            double k_modulus {std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])};
            std::array<double, 3> k_n {k[0]/k_modulus, k[1]/k_modulus, k[2]/k_modulus};


            // build LK Hamiltonian matrix (3x3)
            Eigen::Matrix3d LK_matrix;
            LK_matrix << A_LK*k_n[0]*k_n[0] + B_LK*(k_n[1]*k_n[1] + k_n[2]*k_n[2]), C_LK*k_n[0]*k_n[1], C_LK*k_n[0]*k_n[2],
                         C_LK*k_n[0]*k_n[1], A_LK*k_n[1]*k_n[1] + B_LK*(k_n[0]*k_n[0] + k_n[2]*k_n[2]), C_LK*k_n[1]*k_n[2],
                         C_LK*k_n[0]*k_n[2], C_LK*k_n[1]*k_n[2], A_LK*k_n[2]*k_n[2] + B_LK*(k_n[0]*k_n[0] + k_n[1]*k_n[1]);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver; 
            eigensolver.compute(LK_matrix, Eigen::EigenvaluesOnly); // compute eigenvalues

            if(eigensolver.info() != 0){
                const double nan {std::numeric_limits<double>::quiet_NaN()};
                eigenvalues << nan, nan, nan;
                return eigenvalues; // solver failure sentinel (NaN, never a valid eigenvalue)
            }

            eigenvalues = eigensolver.eigenvalues().transpose(); // eigenvalues row vector (from smallest to largest)

            return eigenvalues;
        }

        // electron effective mass from LK eigenvalue
        static double computeEffMassfromEigenval(double eigenval){
            return 1/(2*eigenval);
        }

        static std::array<double, 3> computeEffMassfromEigenval(std::array<double, 3> eigenval){
            return std::array<double, 3> {1/(2*eigenval[0]), 1/(2*eigenval[1]), 1/(2*eigenval[2])};
        }
    
        static void computeRightSide(Vertex * left_most, Vertex * right_most){
            assert(left_most != nullptr);
            assert(right_most != nullptr);

            Vertex * ptr {right_most->prev};

            // seed with the suffix starting at right_most, so a partial refresh (right_most short
            // of the tail) carries the untouched rest of the diagram in instead of dropping it:
            // R(right_most->prev) = A(right_most->prev) * wf(right_most) * R(right_most). The tail
            // itself contributes nothing, hence Identity there. right_most's own wf and
            // right_component must already be current when this is called with right_most != tail.
            Eigen::Matrix3d weight_right {
                right_most->next == nullptr ? Eigen::Matrix3d::Identity()
                                            : Eigen::Matrix3d(right_most->vertex_wf_component * right_most->right_component)
            };

            while (ptr != left_most) {
                weight_right =  ptr->el_prop_action.diagonal().asDiagonal() * weight_right;
        
                ptr->right_component = weight_right;

                weight_right = ptr->vertex_wf_component * weight_right;

                ptr = ptr->prev;
            }

            weight_right = ptr->el_prop_action.diagonal().asDiagonal() * weight_right;

            ptr->right_component = weight_right;
        }

        static void computeLeftSide(Vertex * right_most, Vertex * left_most){
            assert(right_most != nullptr);
            assert(left_most != nullptr);

            Vertex * ptr {left_most};

            // seed with the prefix through left_most, so a partial refresh (left_most past the
            // head) carries the untouched start of the diagram in instead of dropping it:
            // L(left_most->next) = L(left_most) * wf(left_most) * A(left_most). The head has no
            // prefix and no incoming overlap, hence just its action there. left_most's own
            // left_component and wf must already be current when this is called with
            // left_most != head; its el_prop_action is read fresh either way.
            Eigen::Matrix3d weight_left {
                left_most->prev == nullptr
                    ? Eigen::Matrix3d(left_most->el_prop_action.diagonal().asDiagonal())
                    : Eigen::Matrix3d(left_most->left_component * left_most->vertex_wf_component * left_most->el_prop_action.diagonal().asDiagonal())
            };

            ptr = ptr->next;

            while (ptr != right_most) {
                ptr->left_component = weight_left;

                weight_left = weight_left * ptr->vertex_wf_component * ptr->el_prop_action.diagonal().asDiagonal();

                ptr = ptr->next;
            }

            ptr->left_component = weight_left;
        }
    };
}

#endif // !WEIGHT_COMPUTATION_HPP
