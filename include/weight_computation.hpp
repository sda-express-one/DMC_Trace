#ifndef WEIGHT_COMPUTATION_HPP
#define WEIGHT_COMPUTATION_HPP

#include <Eigen/Core>
#include <cassert>
#include "vertex.hpp"

namespace weight {
    namespace phononic {

    }

    namespace LKMatrix {
        inline void computeRightSide(Vertex * left_most, Vertex * diagram_tail){
            assert(left_most != nullptr);
            assert(diagram_tail != nullptr);

            Vertex * ptr {diagram_tail->prev};

            Eigen::Matrix3d weight_right {Eigen::Matrix3d::Identity()};

            while (ptr != left_most->next) {
                weight_right =  ptr->vertex_wf_component * ptr->el_prop_action.diagonal().asDiagonal() * weight_right;

                ptr->right_component = weight_right;

                ptr = ptr->prev;
            }

            weight_right = ptr->el_prop_action.diagonal().asDiagonal() * weight_right;

            ptr->right_component = weight_right;
        }

        inline void computeLeftSide(Vertex * right_most, Vertex * diagram_head){
            assert(right_most != nullptr);
            assert(diagram_head != nullptr);

            Vertex * ptr {diagram_head->next};

            Eigen::Matrix3d weight_left {Eigen::Matrix3d::Identity()};

            while (ptr != right_most->prev) {
                weight_left = weight_left * ptr->vertex_wf_component * ptr->el_prop_action.diagonal().asDiagonal();

                ptr->left_component = weight_left;

                ptr = ptr->next;
            }

            weight_left = weight_left * ptr->el_prop_action.diagonal().asDiagonal();

            ptr->left_component = weight_left;
        }
    }
}

#endif // !WEIGHT_COMPUTATION_HPP
