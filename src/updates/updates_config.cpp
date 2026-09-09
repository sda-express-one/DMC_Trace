#include "../../include/updates/updates_config.hpp"
#include <cassert>

updates_cfg::updates_cfg(
    std::array<double, 3> k_init,
    double tau_max, 
    double chem_pot, 
    VertexPointerManager * internal_ph_manager,
    VertexPointerManager * external_ph_manager
    ) : 
        tau_max(tau_max), 
        chem_pot(chem_pot), 
        internal_ph_manager(internal_ph_manager), 
        external_ph_manager(external_ph_manager), 
        max_order_int(internal_ph_manager->max_length),
        max_order_ext(external_ph_manager->max_length)
{
    assert(this->chem_pot < 0);
    assert(internal_ph_manager != nullptr);
    assert(external_ph_manager != nullptr);

    const int max_vertices {max_order_int + max_order_ext + 2};
    assert(max_vertices >= 2);

    // single fixed-size allocation: every element's address is stable for the
    // lifetime of this object, so prev/next pointers into it never dangle.
    vertex_pool = new Vertex[max_vertices];
    for (int i {0}; i < max_vertices; ++i) {
        vertex_pool[i].k = k_init;
        vertex_pool[i].prev = (i > 0) ? &vertex_pool[i - 1] : nullptr;
        vertex_pool[i].next = (i < max_vertices - 1) ? &vertex_pool[i + 1] : nullptr;
    }

    // pop the first two elements off the pool to seed the (initially empty) diagram
    diagram_head = &vertex_pool[0];
    diagram_tail = &vertex_pool[1];
    free_stack = (max_vertices > 2) ? &vertex_pool[2] : nullptr;
    if (free_stack != nullptr) {
        free_stack->prev = nullptr;
    }

    diagram_head->prev = nullptr;
    diagram_head->next = diagram_tail;
    diagram_tail->prev = diagram_head;
    diagram_tail->next = nullptr;
};

updates_cfg::~updates_cfg(){
    delete[] vertex_pool;
}

updates_cfg::updates_cfg(updates_cfg&& other) noexcept :
    vertex_pool(other.vertex_pool),
    free_stack(other.free_stack),
    diagram_head(other.diagram_head),
    diagram_tail(other.diagram_tail),
    internal_ph_manager(other.internal_ph_manager),
    external_ph_manager(other.external_ph_manager),
    max_order_int(other.max_order_int),
    max_order_ext(other.max_order_ext),
    tau_max(other.tau_max),
    chem_pot(other.chem_pot)
{
    other.vertex_pool = nullptr;
    other.free_stack = nullptr;
    other.diagram_head = nullptr;
    other.diagram_tail = nullptr;
    other.internal_ph_manager = nullptr;
    other.external_ph_manager = nullptr;
}
