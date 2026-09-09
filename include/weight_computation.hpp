#ifndef WEIGHT_COMPUTATION_HPP
#define WEIGHT_COMPUTATION_HPP

#include <Eigen/Core>
#include <Eigen/src/Core/util/XprHelper.h>
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

            while (ptr != left_most) {
                weight_right =  ptr->el_prop_action.diagonal().asDiagonal() * weight_right;
        
                ptr->right_component = weight_right;

                weight_right = ptr->vertex_wf_component * weight_right;

                ptr = ptr->prev;
            }

            weight_right = ptr->el_prop_action.diagonal().asDiagonal() * weight_right;

            ptr->right_component = weight_right;
        }

        inline void computeLeftSide(Vertex * right_most, Vertex * diagram_head){
            assert(right_most != nullptr);
            assert(diagram_head != nullptr);

            Vertex * ptr {diagram_head};

            Eigen::Matrix3d weight_left {diagram_head->el_prop_action.diagonal().asDiagonal()};

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
