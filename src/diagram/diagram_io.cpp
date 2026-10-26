#include "diagram/diagram_io.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>
#include "diagram/diagram_builder.hpp"
#include "diagram/phonon_manager.hpp"
#include "diagram/vertex_manager.hpp"

namespace {
    int modeIndex(const diagram_cfg & cfg, const Vertex * v){
        const auto & pool {cfg.phonon_mode_manager->phonon_mode_pool};
        for (std::size_t i {0}; i < pool.size(); ++i) {
            if (pool[i].phonon_energy == v->ph_energy && pool[i].diel_response == v->diel_response) { return static_cast<int>(i); }
        }
        throw std::runtime_error("captureDiagram: a line's phonon mode is not in the mode pool");
    }

    std::vector<LineState> lines(const diagram_cfg & cfg, const VertexPointerManager * manager){
        std::vector<LineState> out;
        for (int i {0}; i < manager->current_length; i += 2) {        // even slot: the creation vertex
            const Vertex * cre {manager->selectVertex(i)};
            out.push_back({cre->tau, cre->conj_vertex->tau, cre->w, modeIndex(cfg, cre)});
        }
        return out;
    }
}

DiagramState captureDiagram(const diagram_cfg & cfg){
    DiagramState d;
    d.tau_D = cfg.current_tau_length;
    d.internal = lines(cfg, cfg.internal_ph_manager);
    d.external = lines(cfg, cfg.external_ph_manager);
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) { d.k.push_back(v->k); }
    return d;
}

numerical::SanitizeReport restoreDiagram(diagram_cfg & cfg, const DiagramState & state){
    diagram_builder::reset(cfg, state.tau_D);
    // appending in the saved order reproduces the managers' slot maps
    for (const LineState & l : state.internal) {
        if (!diagram_builder::addInternalLine(cfg, l.t_cre, l.t_ann, l.w, l.mode)) {
            throw std::runtime_error("restoreDiagram: an internal line could not be placed (capacity or times)");
        }
    }
    for (const LineState & l : state.external) {
        if (!diagram_builder::addExternalLine(cfg, l.t_cre, l.t_ann, l.w, l.mode)) {
            throw std::runtime_error("restoreDiagram: an external line could not be placed (capacity or times)");
        }
    }
    std::size_t n {0};
    for (const Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) { ++n; }
    if (n != state.k.size()) {
        throw std::runtime_error("restoreDiagram: " + std::to_string(state.k.size()) + " saved momenta for " + std::to_string(n) + " segments");
    }
    std::size_t i {0};
    for (Vertex * v {cfg.diagram_head}; v != cfg.diagram_tail; v = v->next) { v->k = state.k[i++]; }
    // without external lines the ends carry exactly p (diagram_cfg::p_ext): repairs states saved before the end
    // segments were kept canonical (a rounding-level change)
    if (cfg.endsAtP()) {
        cfg.diagram_head->k = cfg.p_ext;
        cfg.diagram_tail->prev->k = cfg.p_ext;
    }
    return diagram_builder::finalize(cfg);
}
