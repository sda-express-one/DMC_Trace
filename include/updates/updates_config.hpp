#ifndef UPDATE_CONFIG_HPP
#define UPDATE_CONFIG_HPP

#include <cassert>
#include "../vertex.hpp"
#include "../vertex_manager.hpp"

struct updates_cfg {
    Vertex * diagram_head {nullptr};
    Vertex * diagram_tail {nullptr};

    VertexPointerManager * internal_ph_manager {nullptr};
    VertexPointerManager * external_ph_manager {nullptr};
    
    const double tau_max {50.};
    const double chem_pot {-1.};

    updates_cfg(Vertex * diagram_head, 
            Vertex * diagram_tail, 
            double tau_max = 50.0, 
            double chem_pot = -1.0, 
            VertexPointerManager * internal_ph_manager = nullptr,
            VertexPointerManager * external_ph_manager = nullptr
                
        ) : diagram_head(diagram_head), diagram_tail(diagram_tail), tau_max(tau_max), chem_pot(chem_pot), 
            internal_ph_manager(internal_ph_manager), external_ph_manager(external_ph_manager) {
        assert(this->diagram_tail != nullptr);
        assert(this->chem_pot < 0);     
    }
};

#endif // !UPDATE_CONFIG_HPP
