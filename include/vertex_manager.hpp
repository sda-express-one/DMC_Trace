#ifndef VERTEX_MANAGER_HPP
#define VERTEX_MANAGER_HPP

#include <random>
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
    VertexPointer * ptr_vertex_pool {nullptr}; // fixed-size pool, allocated once: addresses into it
                                                // (via .conjugated) must stay stable for the whole
                                                // lifetime of the manager, hence a raw array rather
                                                // than a std::vector that could reallocate.
    mutable simplemc::xoshiro256ss * rng;

    VertexPointerManager(int max_length, simplemc::xoshiro256ss * rng);
    ~VertexPointerManager();

    // ptr_vertex_pool is uniquely owned (delete[]'d in the destructor), so copying would
    // double-free; only moving it out is allowed. max_length is const, so a move-assignment
    // operator can't reinitialize it - only move-construction is meaningful here.
    VertexPointerManager(const VertexPointerManager&) = delete;
    VertexPointerManager& operator=(const VertexPointerManager&) = delete;
    VertexPointerManager& operator=(VertexPointerManager&&) = delete;

    VertexPointerManager(VertexPointerManager&& other) noexcept;

    void addVertexPointers(Vertex * vertex_one, Vertex * vertex_two){
        assert(vertex_one != nullptr);
        assert(vertex_two != nullptr);
        assert(current_length % 2 == 0);
        assert(current_length < max_length + 1);

        if(current_length + 2 > max_length){
            return;
        }

        const int position_one {current_length};
        const int position_two {current_length + 1};

        ptr_vertex_pool[position_one] = VertexPointer{.linked_vertex = vertex_one, .position = position_one, .used = true};
        ptr_vertex_pool[position_two] = VertexPointer{.linked_vertex = vertex_two, .position = position_two, .used = true};

        // conjugated must point at the pool's own (stable) slots, not at the temporaries above
        ptr_vertex_pool[position_one].conjugated = &ptr_vertex_pool[position_two];
        ptr_vertex_pool[position_two].conjugated = &ptr_vertex_pool[position_one];

        current_length += 2;
    }

    void removeVertexPointers(VertexPointer& pointer_one, VertexPointer& pointer_two){
        assert(current_length % 2 == 0);
        assert(current_length > -1);
        
        if(current_length < 2){
            return;
        }

        if(pointer_one.position != current_length - 2 && pointer_two.position != current_length - 1){
            const int position_one {pointer_one.position};
            const int position_two {pointer_two.position};

            ptr_vertex_pool[position_one].linked_vertex = ptr_vertex_pool[current_length-2].linked_vertex;
            ptr_vertex_pool[position_two].linked_vertex = ptr_vertex_pool[current_length-1].linked_vertex;

            ptr_vertex_pool[position_one].conjugated = &ptr_vertex_pool[position_two];
            ptr_vertex_pool[position_two].conjugated = &ptr_vertex_pool[position_one];
        }

        ptr_vertex_pool[current_length - 1].linked_vertex = nullptr;
        ptr_vertex_pool[current_length - 1].conjugated = nullptr;
        ptr_vertex_pool[current_length - 1].used = false;
        ptr_vertex_pool[current_length - 1].position = -1;

        ptr_vertex_pool[current_length - 2].linked_vertex = nullptr;
        ptr_vertex_pool[current_length - 2].conjugated = nullptr;
        ptr_vertex_pool[current_length - 2].used = false;
        ptr_vertex_pool[current_length - 2].position = -1;

        current_length -= 2;
    }

    Vertex * selectVertex(int position) const {
        assert(position < current_length);
        return ptr_vertex_pool[position].linked_vertex;
    }

    VertexPointer * chooseOutgoingVertex() {
        int type {0};
        int position {-1};
        assert(current_length <= max_length);
        std::uniform_int_distribution<int> select {0, current_length - 1};

        do {
            position = select(*rng);
            assert(ptr_vertex_pool[position].linked_vertex != nullptr);
            type = ptr_vertex_pool[position].linked_vertex->type;
        } while (type < 1);

        return &ptr_vertex_pool[position];
    }
};

#endif // !VERTEX_MANAGER_HPP
