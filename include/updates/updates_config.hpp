#ifndef UPDATE_CONFIG_HPP
#define UPDATE_CONFIG_HPP

#include <cassert>
#include <array>
#include "../vertex.hpp"
#include "../vertex_manager.hpp"

struct updates_cfg {
    Vertex * vertex_pool {nullptr}; // base of the new[]-allocated pool, kept only for cleanup
    Vertex * free_stack {nullptr};  // current head of the still-unused portion of the pool
    Vertex * diagram_head {nullptr};
    Vertex * diagram_tail {nullptr};

    VertexPointerManager * internal_ph_manager {nullptr};
    VertexPointerManager * external_ph_manager {nullptr};

    const int max_order_int {0};
    const int max_order_ext {0};
    
    const double tau_max {50.};
    const double chem_pot {-1.};

    double current_tau_length {1.};

    updates_cfg(
            std::array<double, 3> k_init = {0, 0, 0},
            double tau_max = 50.0,
            double chem_pot = -1.0,
            VertexPointerManager * internal_ph_manager = nullptr,
            VertexPointerManager * external_ph_manager = nullptr
        );
    ~updates_cfg();

    // vertex_pool is uniquely owned (delete[]'d in the destructor), so copying would
    // double-free; only moving it out is allowed. max_order_int/max_order_ext/tau_max/
    // chem_pot are const, so a move-assignment operator can't reinitialize them - only
    // move-construction is meaningful here.
    updates_cfg(const updates_cfg&) = delete;
    updates_cfg& operator=(const updates_cfg&) = delete;
    updates_cfg& operator=(updates_cfg&&) = delete;

    updates_cfg(updates_cfg&& other) noexcept;

    Vertex * findPositionFromLeft(Vertex * left_most, const double tau_pos_to_find){
        assert(left_most !=  nullptr);
        assert(tau_pos_to_find > 0);

        if(tau_pos_to_find < left_most->tau){
            return nullptr;
        }

        Vertex * ptr {left_most};

        while (tau_pos_to_find > ptr->tau_next) {
            ptr = ptr->next;
            assert(ptr != nullptr);
        }
        return ptr;
    }

    Vertex * findPositionFromRight(Vertex * right_most, const double tau_pos_to_find){
        assert(right_most != nullptr);
        assert(tau_pos_to_find > 0);

        if(tau_pos_to_find > right_most->tau_next){
            return nullptr;
        }

        Vertex * ptr {right_most};

        while (tau_pos_to_find < ptr->tau) {
            ptr = ptr->prev;
            assert(ptr != nullptr);
        }
        return ptr;
    }

    Vertex * drawVertexFromPool(){
        assert(free_stack != nullptr);
        Vertex * vertex_to_draw {free_stack};
        free_stack = free_stack->next;
        if (free_stack != nullptr) {
            free_stack->prev = nullptr;
        }
        vertex_to_draw->next = nullptr;
        return vertex_to_draw;
    }

    void addVertex(Vertex * vertex_to_add, Vertex * vertex_to_l){
        assert(vertex_to_add != nullptr);
        assert(vertex_to_l != nullptr);

        Vertex * vertex_to_r {vertex_to_l->next};
        assert(vertex_to_r != nullptr);

        vertex_to_r->prev = nullptr;
        vertex_to_l->next = nullptr;
        vertex_to_add->prev = vertex_to_l;
        vertex_to_l->next = vertex_to_add;
        vertex_to_add->next = vertex_to_r;
        vertex_to_r->prev = vertex_to_add;
    }
    
    Vertex * removeVertex(Vertex * vertex_to_rm){
        assert(vertex_to_rm != nullptr);

        Vertex * vertex_to_l {vertex_to_rm->prev};
        assert(vertex_to_l != nullptr);
        Vertex * vertex_to_r {vertex_to_rm->next};
        assert(vertex_to_r != nullptr);
        
        vertex_to_l->next = vertex_to_r;
        vertex_to_r->prev = vertex_to_l;

        vertex_to_rm->prev = nullptr;
        vertex_to_rm->next = nullptr;

        return vertex_to_rm;
    }
    
    void addVertexToPool(Vertex * vertex_to_pool){
        assert(vertex_to_pool != nullptr);
        assert(vertex_to_pool->prev == nullptr);
        assert(vertex_to_pool->next == nullptr);

        vertex_to_pool->next = free_stack;
        if (free_stack != nullptr) {
            free_stack->prev = vertex_to_pool;
        }
        free_stack = vertex_to_pool;
    }
};

#endif // !UPDATE_CONFIG_HPP
