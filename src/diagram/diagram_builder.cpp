#include "diagram/diagram_builder.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include "diagram/phonon_manager.hpp"
#include "diagram/vertex_manager.hpp"
#include "utils/prop_distribs.hpp"

namespace diagram_builder {

    namespace {
        // true iff two more vertices can be drawn from the pool
        bool poolHasTwo(const diagram_cfg & cfg){
            return cfg.free_stack != nullptr && cfg.free_stack->next != nullptr;
        }

        // the segment a new vertex at time t splits, or nullptr if t is outside (0, tau_D) or ties with a vertex
        Vertex * segmentFor(diagram_cfg & cfg, double t){
            if (!(t > 0. && t < cfg.current_tau_length)) { return nullptr; }
            Vertex * seg {cfg.findPositionFromLeft(cfg.diagram_head, t)};
            return (seg != nullptr && diagram_cfg::strictlyInside(seg, t)) ? seg : nullptr;
        }

        // a vertex drawn from the pool, linked after seg at time t (seg must have been checked by segmentFor)
        Vertex * insertAt(diagram_cfg & cfg, Vertex * seg, double t){
            Vertex * v {cfg.drawVertexFromPool()};
            v->tau = t;
            v->tau_next = seg->tau_next;
            seg->tau_next = t;
            cfg.addVertex(v, seg);
            return v;
        }

        bool addLine(diagram_cfg & cfg, bool internal, double t_cre, double t_ann, const std::array<double, 3> & w, int mode){
            VertexPointerManager * manager {internal ? cfg.internal_ph_manager : cfg.external_ph_manager};
            if (manager->current_length + 2 > manager->max_length || !poolHasTwo(cfg)) { return false; }
            if (mode < 0 || mode >= cfg.phonon_mode_manager->num_phonon_modes) { return false; }
            if (t_cre == t_ann || (internal && !(t_cre < t_ann))) { return false; }

            // both positions are checked before anything is inserted, so a failure leaves the diagram as it was;
            // two distinct times strictly inside segments stay strictly inside after either is inserted
            Vertex * seg_cre {segmentFor(cfg, t_cre)};
            Vertex * seg_ann {segmentFor(cfg, t_ann)};
            if (seg_cre == nullptr || seg_ann == nullptr) { return false; }

            // insert the later time first, so the earlier one's segment pointer stays the one that contains it
            const bool cre_first {t_cre < t_ann};
            Vertex * late {insertAt(cfg, cre_first ? seg_ann : seg_cre, cre_first ? t_ann : t_cre)};
            Vertex * early {insertAt(cfg, cfg.findPositionFromLeft(cfg.diagram_head, cre_first ? t_cre : t_ann),
                                     cre_first ? t_cre : t_ann)};
            Vertex * cre {cre_first ? early : late};
            Vertex * ann {cre_first ? late : early};

            const PhononMode & pm {cfg.phonon_mode_manager->phonon_mode_pool[static_cast<std::size_t>(mode)]};
            const int kind {internal ? 1 : 2};
            cre->type = +kind;
            ann->type = -kind;
            cre->w = w;
            ann->w = w;
            cre->ph_energy = ann->ph_energy = pm.phonon_energy;
            cre->diel_response = ann->diel_response = pm.diel_response;
            cre->conj_vertex = ann;
            ann->conj_vertex = cre;
            manager->addVertexPointers(cre, ann);       // creation first: the manager's slot layout
            return true;
        }
    }

    void reset(diagram_cfg & cfg, double tau_D){
        assert(tau_D > 0.);
        for (VertexPointerManager * manager : {cfg.internal_ph_manager, cfg.external_ph_manager}) {
            while (manager->current_length >= 2) {
                // the last pair: removing it moves nothing else
                VertexPointer & one {manager->ptr_vertex_pool[manager->current_length - 2]};
                VertexPointer & two {manager->ptr_vertex_pool[manager->current_length - 1]};
                Vertex * a {one.linked_vertex};
                Vertex * b {two.linked_vertex};
                manager->removeVertexPointers(one, two);
                for (Vertex * v : {a, b}) {
                    cfg.removeVertex(v);
                    v->type = 0;
                    v->index = -1;
                    v->conj_vertex = nullptr;
                    cfg.addVertexToPool(v);
                }
            }
        }
        assert(cfg.diagram_head->next == cfg.diagram_tail);

        cfg.current_tau_length = tau_D;
        cfg.diagram_head->tau = 0.;
        cfg.diagram_head->tau_next = tau_D;
        cfg.diagram_tail->tau = tau_D;
    }

    bool addInternalLine(diagram_cfg & cfg, double t_cre, double t_ann, const std::array<double, 3> & w, int mode){
        return addLine(cfg, true, t_cre, t_ann, w, mode);
    }

    bool addExternalLine(diagram_cfg & cfg, double t_cre, double t_ann, const std::array<double, 3> & w, int mode){
        return addLine(cfg, false, t_cre, t_ann, w, mode);
    }

    void assignMomenta(diagram_cfg & cfg, const std::array<double, 3> & p){
        std::array<double, 3> k {p};
        const VertexPointerManager * ext {cfg.external_ph_manager};
        for (int i {0}; i < ext->current_length; i += 2) {     // one vertex per external line
            const std::array<double, 3> & w {ext->selectVertex(i)->w};
            for (int a {0}; a < 3; ++a) { k[a] -= w[a]; }
        }
        cfg.diagram_head->k = k;
        for (Vertex * v {cfg.diagram_head->next}; v != cfg.diagram_tail; v = v->next) {
            const double sgn {v->type > 0 ? 1. : -1.};
            for (int a {0}; a < 3; ++a) { k[a] -= sgn * v->w[a]; }
            v->k = k;
        }
        // the two ends carry the same momentum, p - sum of the external w: taken from the head bitwise (the running
        // sum above reaches it only up to rounding), so both ends get the same basis and, with no external line, the
        // last segment is exactly p (diagram_cfg::p_ext)
        cfg.diagram_tail->prev->k = cfg.diagram_head->k;
    }

    numerical::SanitizeReport finalize(diagram_cfg & cfg){
        for (const VertexPointerManager * manager : {cfg.internal_ph_manager, cfg.external_ph_manager}) {
            for (int i {0}; i < manager->current_length; ++i) { manager->selectVertex(i)->vertexStrength(); }
        }
        // the first pass compares against the stale cache (always "dirty") and rebuilds it; the second checks
        // the rebuilt diagram, so its report is the meaningful one
        numerical::sanitizeDiagram(&cfg);
        return numerical::sanitizeDiagram(&cfg);
    }

    numerical::SanitizeReport randomDiagram(diagram_cfg & cfg, simplemc::xoshiro256ss & rng, const std::array<double, 3> & p,
                                            double tau_D, const RandomDiagramSpec & spec){
        assert(spec.n_internal_min >= 0 && spec.n_internal_min <= spec.n_internal_max);
        assert(spec.n_external_min >= 0 && spec.n_external_min <= spec.n_external_max);

        // A random diagram can have a weight too small to represent: the cached trace stores band-normalised
        // actions (each segment's lowest band = 1), and when different bands are lowest on different segments
        // (e.g. C = 0 with large momenta) every band's product can underflow to 0. Such a start is unusable
        // (every ratio divides by it), and a chain would never visit it: the whole diagram is redrawn.
        numerical::SanitizeReport rep;
        for (int attempt {0}; attempt < max_attempts; ++attempt) {
            rep = randomDiagramOnce(cfg, rng, p, tau_D, spec);
            const double tr {cfg.diagram_head->right_component.trace()};
            if (rep.clean() && std::isfinite(tr) && tr != 0.) { break; }
        }
        return rep;
    }

    numerical::SanitizeReport randomDiagramOnce(diagram_cfg & cfg, simplemc::xoshiro256ss & rng, const std::array<double, 3> & p,
                                                double tau_D, const RandomDiagramSpec & spec){
        reset(cfg, tau_D);

        // capacity: lines the managers and the pool can hold (two vertices each, head and tail excluded)
        const int pool_lines {(cfg.max_vertices - 2) / 2};
        const int max_int {std::min(cfg.internal_ph_manager->max_length / 2, pool_lines)};
        std::uniform_int_distribution<int> draw_int {std::min(spec.n_internal_min, max_int), std::min(spec.n_internal_max, max_int)};
        const int n_int {draw_int(rng)};
        const int max_ext {std::min(cfg.external_ph_manager->max_length / 2, pool_lines - n_int)};
        std::uniform_int_distribution<int> draw_ext {std::min(spec.n_external_min, max_ext), std::min(spec.n_external_max, max_ext)};
        const int n_ext {draw_ext(rng)};

        assert(spec.w_spread > 0.);
        std::uniform_real_distribution<double> draw_t {0., tau_D};
        std::uniform_real_distribution<double> unif {0., 1.};
        std::uniform_int_distribution<int> draw_mode {0, cfg.phonon_mode_manager->num_phonon_modes - 1};
        const proposal::PhononMomentum w_proposal {.shape = 1. / (spec.w_spread * spec.w_spread)};
        const auto draw_exp {[&](double omega){ return -std::log(1. - unif(rng)) / omega; }};

        // times are redrawn while the line does not fit (beyond tau_D, or creation first for an external line) or
        // ties with an existing vertex; a line that does not fit in max_line_draws draws (tau_D << 1/omega) is left out
        constexpr int max_line_draws {100000};
        for (int n {0}; n < n_int + n_ext; ++n) {
            const bool internal {n < n_int};
            const int mode {draw_mode(rng)};
            const double omega {cfg.phonon_mode_manager->phonon_mode_pool[static_cast<std::size_t>(mode)].phonon_energy};
            for (int draw {0}; draw < max_line_draws; ++draw) {
                if (internal) {
                    const double t_cre {draw_t(rng)}, t_ann {t_cre + draw_exp(omega)};
                    if (!(t_ann < tau_D)) { continue; }
                    if (addInternalLine(cfg, t_cre, t_ann, w_proposal.draw(rng, t_ann - t_cre), mode)) { break; }
                }
                else {
                    const double t_ann {draw_exp(omega)}, t_cre {tau_D - draw_exp(omega)};
                    if (!(t_ann < t_cre)) { continue; }
                    if (addExternalLine(cfg, t_cre, t_ann, w_proposal.draw(rng, tau_D - t_cre + t_ann), mode)) { break; }
                }
            }
        }

        assignMomenta(cfg, p);
        return finalize(cfg);
    }
}
