#ifndef VERTEX_MANAGER_HPP
#define VERTEX_MANAGER_HPP

#include <random>
#include <vector>
#include <cassert>
#include <simplemc/random/xoshiro256.hpp>
#include "vertex.hpp"

struct VertexPointer {
    Vertex * linked_vertex {nullptr};
    VertexPointer * conjugated {nullptr};
    int position {-1};
    bool used {false};
};

struct VertexPointerManager {
    const int max_length {0};
    int current_length {0};
    std::vector<VertexPointer> ptr_vertex_vector;
    mutable simplemc::xoshiro256ss * rng;

    VertexPointerManager(int max_length, simplemc::xoshiro256ss * rng) : max_length(max_length), rng(rng) {
        ptr_vertex_vector.reserve(this->max_length);
    }

    void addVertexPointers(Vertex * vertex_one, Vertex * vertex_two){
        assert(vertex_one != nullptr);
        assert(vertex_two != nullptr);
        assert(current_length % 2 == 0);
        assert(current_length < max_length);

        if(current_length + 2 > max_length){
            return;
        }

        VertexPointer pointer_one {.linked_vertex = vertex_one, .position = current_length, .used = true};
        VertexPointer pointer_two {.linked_vertex = vertex_two, .position = current_length + 1, .used = true};
        pointer_one.conjugated = &pointer_two;
        pointer_two.conjugated = &pointer_one;

        ptr_vertex_vector.push_back(pointer_one);
        ptr_vertex_vector.push_back(pointer_two);

        current_length += 2;
    }
    
    void removeVertexPointers(VertexPointer& pointer_one, VertexPointer& pointer_two){
        assert(current_length % 2 == 0);
        assert(current_length > 0);

        if(pointer_one.position != current_length - 2 && pointer_two.position != current_length - 1){
            int position_one {pointer_one.position};
            int position_two {pointer_two.position};

            ptr_vertex_vector[position_one].linked_vertex = ptr_vertex_vector[current_length-2].linked_vertex;
            ptr_vertex_vector[position_two].linked_vertex = ptr_vertex_vector[current_length-1].linked_vertex;

            ptr_vertex_vector[position_one].conjugated = &ptr_vertex_vector[position_two];
            ptr_vertex_vector[position_two].conjugated = &ptr_vertex_vector[position_one];
        }
        
        ptr_vertex_vector[current_length - 1].linked_vertex = nullptr;
        ptr_vertex_vector[current_length - 1].conjugated = nullptr;
        ptr_vertex_vector[current_length - 1].used = false;
        ptr_vertex_vector[current_length - 1].position = -1;

        ptr_vertex_vector[current_length - 2].linked_vertex = nullptr;
        ptr_vertex_vector[current_length - 2].conjugated = nullptr;
        ptr_vertex_vector[current_length - 2].used = false;
        ptr_vertex_vector[current_length - 2].position = -1;

        current_length -= 2;
    }

    Vertex * selectVertex(int position) const {
        assert(position < current_length);
        return ptr_vertex_vector[position].linked_vertex;
    }

    VertexPointer * chooseOutgoingNode() {
        int type {0};
        int position {-1};
        assert(current_length == ptr_vertex_vector.size());
        std::uniform_int_distribution<int> select {0, current_length - 1};

        do {
            position = select(*rng);
            assert(ptr_vertex_vector[position].linked_vertex != nullptr);
            type = ptr_vertex_vector[position].linked_vertex->type;
        } while (type < 1);

        return &ptr_vertex_vector[position];
    }
};

#endif // !VERTEX_MANAGER_HPP
