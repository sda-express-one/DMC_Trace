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
        struct choice {
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

        // Column n: the signs (+-1, or 0 for a component that is exactly 0, only the C = 0 axes) of band n's
        // eigenvector in the interior of the wedge k_0 > k_1 > k_2 > 0, where they are constant (see the sign gauge in
        // diagonalizeLKHamiltonian). Set by setLKParameters; the identity is the pattern of the defaults A = B, C = 0.
        inline static Eigen::Matrix3d sign_pattern {Eigen::Matrix3d::Identity()};

        static void setLKParameters(double A_LK, double B_LK, double C_LK){
            weight::LKMatrix::A_LK = A_LK;
            weight::LKMatrix::B_LK = B_LK;
            weight::LKMatrix::C_LK = C_LK;
            updateSignPattern();
        }

        // Relative tolerance on the direction k_hat below which diagonalizeLKHamiltonian treats a component as 0, or two
        // magnitudes as equal (canonicalDirection).
        inline static constexpr double gauge_tol {1e-9};

        // k_hat = k / |k|, with every direction within gauge_tol of a symmetry plane or line mapped to one canonical
        // value, signs kept. The gauge fixing below is discontinuous exactly there (the sign of a component, the order
        // of the wedge sort, which of the two wedges next to a mirror plane its points are reduced from, and the solver's free rotation in
        // the degenerate pair on (100) and (111)), so two momenta that are equal physically but differ by rounding
        // (a cached k against one rebuilt as (k - w) + w) could get different bases; with the canonical input they
        // get bitwise the same one, on every line of the diagram:
        //   - |k_hat_i| < gauge_tol becomes +0 (never -0: the sign rule must see "not negative");
        //   - magnitudes closer than gauge_tol become equal (their mean), so the wedge sort falls back to index order;
        //   - (100), (110), (111) get the exact magnitudes 1, 1/sqrt(2), 1/sqrt(3): there the input must be bitwise
        //     identical, not only equal up to rounding, since the degenerate pair's eigenvectors are arbitrary;
        //   - any other snapped direction is renormalised (tied entries stay bitwise equal, zeros stay 0).
        // A generic direction (no component below tol, no tie) is the plain k / |k|, bitwise. The snap moves k_hat by
        // at most ~gauge_tol, a relative change of that order in the energies of those directions only.
        // Not covered: a band structure degenerate at every k (C = A - B with C != 0), where the solver's rotation
        // is free everywhere and only bitwise-equal momenta are safe.
        static std::array<double, 3> canonicalDirection(const std::array<double, 3> & k){
            const double n {std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])};
            std::array<double, 3> d {k[0]/n, k[1]/n, k[2]/n};
            std::array<double, 3> m {std::abs(d[0]), std::abs(d[1]), std::abs(d[2])};
            bool changed {false};
            for (int i {0}; i < 3; ++i) {
                if (m[i] < gauge_tol && d[i] != 0.) { m[i] = 0.; changed = true; }
                else if (m[i] < gauge_tol) { m[i] = 0.; }       // also turns -0 into +0
            }

            std::array<int, 3> o {0, 1, 2};     // magnitudes descending
            std::stable_sort(o.begin(), o.end(), [&m](int a, int b){ return m[a] > m[b]; });
            const int nonzero {(m[o[0]] > 0.) + (m[o[1]] > 0.) + (m[o[2]] > 0.)};
            const bool tie01 {m[o[1]] > 0. && m[o[0]] - m[o[1]] < gauge_tol};
            const bool tie12 {m[o[2]] > 0. && m[o[1]] - m[o[2]] < gauge_tol};

            if (nonzero == 1) {
                changed = changed || m[o[0]] != 1.;
                m[o[0]] = 1.;
            }
            else if (nonzero == 2 && tie01) {
                static const double r2 {std::sqrt(0.5)};
                changed = true;
                m[o[0]] = m[o[1]] = r2;
            }
            else if (nonzero == 3 && tie01 && tie12) {
                static const double r3 {std::sqrt(1./3.)};
                changed = true;
                m[o[0]] = m[o[1]] = m[o[2]] = r3;
            }
            else {
                if (tie01 && m[o[0]] != m[o[1]]) { m[o[0]] = m[o[1]] = 0.5*(m[o[0]] + m[o[1]]); changed = true; }
                if (tie12 && m[o[1]] != m[o[2]]) { m[o[1]] = m[o[2]] = 0.5*(m[o[1]] + m[o[2]]); changed = true; }
                if (changed) {
                    const double s {1./std::sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2])};
                    for (double & x : m) { x *= s; }
                }
            }
            if (!changed) { return d; }
            for (int i {0}; i < 3; ++i) { d[i] = d[i] < 0. ? -m[i] : m[i]; }
            return d;
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

            // the direction, snapped onto the symmetry planes and lines it lies on up to rounding (see there): both the
            // wedge reduction and the Hamiltonian are built from it
            const std::array<double, 3> k_hat {canonicalDirection(k)};

            choice eigenv_gauge = selectionRulesLK(k_hat); // cast k-values into IBZ

            double k_vector[3];
            k_vector[0] = std::abs(k_hat[eigenv_gauge.position[0]]);
            k_vector[1] = std::abs(k_hat[eigenv_gauge.position[1]]);
            k_vector[2] = std::abs(k_hat[eigenv_gauge.position[2]]);

            Eigen::Matrix3d eigenvectors;
            Eigen::RowVector3d eigenvalues;
            if(!wedgeEigen(k_vector, eigenvectors, eigenvalues)){
                const double nan {std::numeric_limits<double>::quiet_NaN()};
                result << nan, nan, nan,
                          nan, nan, nan,
                          nan, nan, nan,
                          nan, nan, nan;
                return result; // solver failure sentinel (NaN, never a valid eigenvalue)
            }

            // sign gauge: band n's eigenvector is given the sign of its interior sign pattern, sign_pattern.col(n).
            // In the wedge interior k_0 > k_1 > k_2 > 0 no component of a band can vanish (for C != 0, A - B): u_i = 0
            // forces C k_i (k . u) = 0, so u = e_i x k, and the remaining two rows then require
            // (A - B - C)(k_j^2 - k_k^2) = 0, i.e. a boundary or the spherical point. A conical degeneracy inside
            // would flip u around it, so there is none either. Each band therefore keeps one sign pattern over the
            // whole interior, and u . s = sum |u_i| > 0 there: the rule is the earlier "reference component
            // positive" one (sign_pattern is taken from it), bitwise. On a boundary the components that survive keep
            // the interior signs, so u . s > 0 picks the limit from the interior, with no tolerance: noise in a
            // component that vanishes by symmetry cannot decide the sign. Back in the original frame, U(k) on a
            // boundary is thus the continuous extension from the wedge that the reduction g maps onto this one; the
            // two wedges next to a mirror plane differ only in the sign of the band odd under that mirror. Not
            // covered: the degenerate pair on (100) and (111), which has no interior limit (it depends on the
            // direction of approach) and stays the solver's choice, deterministic for the canonical input; and
            // C = A - B, degenerate at every k.
            for (int n {0}; n < 3; ++n) {
                if(eigenvectors.col(n).dot(sign_pattern.col(n)) < 0.){ eigenvectors.col(n) *= -1; }
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

        // H_LK at the wedge direction k_vector (k_0 >= k_1 >= k_2 >= 0, unit length), diagonalised, without the sign
        // gauge; false on a solver failure
        static bool wedgeEigen(const double k_vector[3], Eigen::Matrix3d & eigenvectors, Eigen::RowVector3d & eigenvalues){
            // build LK Hamiltonian matrix (3x3)
            Eigen::Matrix3d LK_matrix;
            LK_matrix << A_LK*k_vector[0]*k_vector[0] + B_LK*(k_vector[1]*k_vector[1] + k_vector[2]*k_vector[2]), C_LK*k_vector[0]*k_vector[1], C_LK*k_vector[0]*k_vector[2],
                     C_LK*k_vector[0]*k_vector[1], A_LK*k_vector[1]*k_vector[1] + B_LK*(k_vector[0]*k_vector[0] + k_vector[2]*k_vector[2]), C_LK*k_vector[1]*k_vector[2],
                     C_LK*k_vector[0]*k_vector[2], C_LK*k_vector[1]*k_vector[2], A_LK*k_vector[2]*k_vector[2] + B_LK*(k_vector[0]*k_vector[0] + k_vector[1]*k_vector[1]);

            // Bands are always returned in ascending eigenvalue order: band 0 has the smallest 1/(2m), i.e. it
            // is the heaviest band and the lowest in energy, whatever A, B and C are.
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
                if(eigensolver.info() != 0){ return false; }

                eigenvectors = eigensolver.eigenvectors(); // eigenvector matrix
                eigenvalues = eigensolver.eigenvalues().transpose(); // eigenvalues row vector (from smallest to largest)
            }
            return true;
        }

        // sign_pattern for the current A, B, C: the eigenvector signs at the interior direction (3, 2, 1)/sqrt(14), in
        // the gauge "reference component positive" (reference axis n of band n for A <= B, 2 - n for A > B: the bands
        // lie mostly along axes 0, 1, 2, resp. 2, 1, 0; inside the wedge that component is never 0). An exactly zero
        // component (C = 0) keeps 0. At C = A - B every k is degenerate and the pattern is the solver's choice.
        static void updateSignPattern(){
            static const double n14 {std::sqrt(14.)};
            const double k_vector[3] {3./n14, 2./n14, 1./n14};
            Eigen::Matrix3d eigenvectors;
            Eigen::RowVector3d eigenvalues;
            if(!wedgeEigen(k_vector, eigenvectors, eigenvalues)){ sign_pattern.setIdentity(); return; }
            const bool reversed {A_LK > B_LK};
            for (int n {0}; n < 3; ++n) {
                if(eigenvectors(reversed ? 2 - n : n, n) < 0){ eigenvectors.col(n) *= -1; }
                for (int i {0}; i < 3; ++i) {
                    sign_pattern(i, n) = std::abs(eigenvectors(i, n)) < 1e-12 ? 0. : (eigenvectors(i, n) > 0. ? 1. : -1.);
                }
            }
        }

        // diagonalizes LK Hamiltonian and returns eigenvalues (no eigenvectors), for effective mass exact estimator
        static Eigen::RowVector3d diagonalizeLKHamiltonianEigenval(const std::array<double, 3> k){

            Eigen::RowVector3d eigenvalues;
        
            // free propagator: same degenerate eigenvalues as diagonalizeLKHamiltonian's k=(0,0,0)
            // case - H_LK vanishes at the extremum, so nothing is split, and 1 rather than 0 only
            // because the caller inverts it for the mass. See there for the full reasoning.
            // Away from k = 0 this function returns the solver's ascending order, the same band order
            // as diagonalizeLKHamiltonian. It uses the plain k / |k|, not canonicalDirection: without
            // eigenvectors there is no gauge to keep stable, and the eigenvalues differ from
            // diagonalizeLKHamiltonian's by at most ~gauge_tol (relative), only next to a symmetry plane.
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
