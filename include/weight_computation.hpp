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
#include "vertex.hpp"
#include "utils/numerical.hpp"

namespace weight {
    struct ProposedVertexWeight {
        std::array<double, 3> k {0., 0., 0.};
        double electron_energy {0.};
        Eigen::Matrix3d el_prop_action {Eigen::Matrix3d::Identity()};
        Eigen::Matrix3d vertex_wf_component {Eigen::Matrix3d::Identity()};
    };

    namespace phononic {

    }

    namespace LKMatrix {
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

        inline static void setLKParameters(double A_LK, double B_LK, double C_LK){
            weight::LKMatrix::A_LK = A_LK;
            weight::LKMatrix::B_LK = B_LK;
            weight::LKMatrix::C_LK = C_LK;
        }

        inline choice selectionRulesLK(const std::array<double, 3> k){
            choice eigenv_gauge; // position starts at the identity {0, 1, 2}

            // sort axis indices by |k| descending; stable_sort keeps the lower original
            // index first on an exact tie, matching the previous >= branch cascade
            std::stable_sort(eigenv_gauge.position.begin(), eigenv_gauge.position.end(),
                              [&k](int a, int b){ return std::abs(k[a]) > std::abs(k[b]); });

            for(int i = 0; i < 3; ++i){
                if(k[eigenv_gauge.position[i]] < 0){ eigenv_gauge.sign[i] = -1; }
            }

            return eigenv_gauge;
        };

        // diagonalizes LK Hamiltonian and returns eigenvalues and eigenvectors in correct order
        inline Eigen::Matrix<double, 4, 3> diagonalizeLKHamiltonian(const std::array<double, 3> k){
    
            Eigen::Matrix<double, 4, 3> result;
    
            // detects free propagators
            if(numerical::isEqual(k[0], 0.) && numerical::isEqual(k[1],0.) && numerical::isEqual(k[2],0.)){
                result << -2, 1, 1,
                          (1./3), 0, 0,
                           0, (1./3), 0,
                           0, 0, (1./3);
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

            Eigen::Matrix3d eigenvectors {eigensolver.eigenvectors()}; // eigenvector matrix 
            Eigen::RowVector3d eigenvalues {eigensolver.eigenvalues().transpose()}; // eigenvalues row vector (from smallest to largest)

            // reorder eigenvectors and eigenvalues (greatest eigenvalue first band for A_LK > B_LK)
            if(A_LK > B_LK){
                double eigenval_temp {0.};
                Eigen::Vector3d column;

                column = eigenvectors.col(2);
                eigenvectors.col(2) = eigenvectors.col(0);
                eigenvectors.col(0) = column;

                eigenval_temp = eigenvalues(0);
                eigenvalues(0) = eigenvalues(2);
                eigenvalues(2) = eigenval_temp;
            }

            // check all diagonal component > 0
            if(eigenvectors(0,0) < 0){eigenvectors.col(0)*=-1;}
            if(eigenvectors(1,1) < 0){eigenvectors.col(1)*=-1;}
            if(eigenvectors(2,2) < 0){eigenvectors.col(2)*=-1;}

            // transformation matrix from IBZ to generic sector
            Eigen::Matrix3d transformation_matrix;
            transformation_matrix << 0, 0, 0,
                                     0, 0, 0,
                                     0, 0, 0;
    
            // compute transformation matrix
            transformation_matrix(eigenv_gauge.position[0],0) = eigenv_gauge.sign[0];
            transformation_matrix(eigenv_gauge.position[1],1) = eigenv_gauge.sign[1];
            transformation_matrix(eigenv_gauge.position[2],2) = eigenv_gauge.sign[2];

            Eigen::Matrix3d new_eigenvectors;

            // new eigenvectord
            Eigen::Vector3d col_zero {transformation_matrix*eigenvectors.col(0)};
            Eigen::Vector3d col_one {transformation_matrix*eigenvectors.col(1)};
            Eigen::Vector3d col_two {transformation_matrix*eigenvectors.col(2)};

            // new eigenvector matrix
            new_eigenvectors.col(0) = col_zero;
            new_eigenvectors.col(1) = col_one;
            new_eigenvectors.col(2) = col_two;

            eigenvectors = new_eigenvectors;

            result << eigenvalues, eigenvectors;

            // return packed result
            return result;
        };

        // diagonalizes LK Hamiltonian and returns eigenvalues (no eigenvectors), for effective mass exact estimator
        inline Eigen::RowVector3d diagonalizeLKHamiltonianEigenval(const std::array<double, 3> k){

            Eigen::RowVector3d eigenvalues;
        
            // free propagator: same well-defined degenerate eigenvalues as diagonalizeLKHamiltonian's k=(0,0,0) case
            if(numerical::isEqual(k[0],0) && numerical::isEqual(k[1],0) && numerical::isEqual(k[2],0)){
                eigenvalues << -2, 1, 1;
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
        };

        // electron effective mass from LK eigenvalue
        inline double computeEffMassfromEigenval(double eigenval){
            return 1/(2*eigenval);
        };

        inline std::array<double, 3> computeEffMassfromEigenval(std::array<double, 3> eigenval){
            return std::array<double, 3> {1/(2*eigenval[0]), 1/(2*eigenval[1]), 1/(2*eigenval[2])};
        }
    
        inline void computeRightSide(Vertex * left_most, Vertex * right_most){
            assert(left_most != nullptr);
            assert(right_most != nullptr);

            Vertex * ptr {right_most->prev};

            Eigen::Matrix3d weight_right {Eigen::Matrix3d::Identity()};

            while (ptr != left_most) {
                weight_right =  ptr->el_prop_action.diagonal().asDiagonal() * weight_right;
        
                ptr->right_component = weight_right;

                weight_right = ptr->vertex_wf_component * weight_right;

                ptr = ptr->prev;
            }

            weight_right = ptr->el_prop_action.diagonal().asDiagonal() * weight_right;

            ptr->right_component = weight_right;
        }

        inline void computeLeftSide(Vertex * right_most, Vertex * left_most){
            assert(right_most != nullptr);
            assert(left_most != nullptr);

            Vertex * ptr {left_most};

            Eigen::Matrix3d weight_left {left_most->el_prop_action.diagonal().asDiagonal()};

            ptr = ptr->next;

            while (ptr != right_most) {
                ptr->left_component = weight_left;

                weight_left = weight_left * ptr->vertex_wf_component * ptr->el_prop_action.diagonal().asDiagonal();

                ptr = ptr->next;
            }

            ptr->left_component = weight_left;
        }
    }
}

#endif // !WEIGHT_COMPUTATION_HPP
