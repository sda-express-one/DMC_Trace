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
};

#endif // !UPDATE_CONFIG_HPP
