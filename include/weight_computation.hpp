#ifndef WEIGHT_COMPUTATION_HPP
#define WEIGHT_COMPUTATION_HPP

#include <Eigen/Core>
#include <cassert>
#include "vertex.hpp"

namespace weight {
    namespace phononic {

    }

    namespace LKMatrix {
        inline Eigen::Matrix3d computeTraceRightSide(Vertex * left_most, Vertex * diagram_tail){
            assert(left_most != nullptr);
            assert(diagram_tail != nullptr);

            Vertex * ptr {diagram_tail->prev};

            Eigen::Matrix3d trace_right {Eigen::Matrix3d::Identity()};

            while (ptr != left_most->next) {
                trace_right =  ptr->vertex_wf_component * ptr->el_prop_action.diagonal().asDiagonal() * trace_right;

                ptr->right_component = trace_right;

                ptr = ptr->prev;
            }

            trace_right = ptr->el_prop_action.diagonal().asDiagonal() * trace_right;

            return trace_right;
        }

        inline Eigen::Matrix3d computeTraceLeftSide(Vertex * right_most, Vertex * diagram_head){
            assert(right_most != nullptr);
            assert(diagram_head != nullptr);

            Vertex * ptr {right_most->prev};

            Eigen::Matrix3d trace_left {Eigen::Matrix3d::Identity()};

            while (ptr != diagram_head) {
                trace_left =  ptr->vertex_wf_component * ptr->el_prop_action.diagonal().asDiagonal() * trace_left;

                ptr->left_component = trace_left;

                ptr = ptr->prev;
            }

            trace_left = ptr->el_prop_action.diagonal().asDiagonal() * trace_left; 

            return trace_left;
        }
    }
}

#endif // !WEIGHT_COMPUTATION_HPP
