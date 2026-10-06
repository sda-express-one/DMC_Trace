#ifndef DIAGRAM_SANITIZER_HPP
#define DIAGRAM_SANITIZER_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <Eigen/Core>
#include "diagram/diagram_config.hpp"
#include "diagram/vertex.hpp"
#include "diagram/vertex_manager.hpp"
#include "comp_method/weight_computation.hpp"

// Periodic integrity check and repair of a diagram's cached state.
//
// Every cached quantity on a vertex is a function of that segment's momentum k and duration:
// baseWF and eff_masses come from diagonalizing H_LK(k) (or are copied together with the k they
// belong to), el_prop_action from k, the masses and tau_next - tau, and vertex_wf_component from
// the two neighbouring bases. In a healthy diagram these all hold exactly. They can drift apart
// only where an update reuses a cached overlap across a basis that was rebuilt separately (e.g.
// rm_int_ph's adjacent case), and then normally by rounding only - unless the diagonalizer's
// gauge fixing (sign convention, IBZ ordering) lands differently for two momenta that are equal
// physically but not bitwise. That is a gauge flip, and it changes the weight at O(1).
//
// sanitizeDiagram() measures every such deviation, then rebuilds all cached state from k and tau
// and refreshes the left/right products over the whole diagram. k itself is treated as the source
// of truth and never modified. Kept out of numerical.hpp because it needs weight_computation.hpp,
// which itself includes numerical.hpp.
namespace numerical {

    struct SanitizeTolerances {
        double exact {1e-12};    // reproduced up to rounding: basis vs U(k), masses, actions (relative)
        double momentum {1e-9};  // conservation across vertices; k accumulates rounding over a run
        double flip {1e-6};      // overlap / closing basis: rounding-level is fine, beyond is a flip
    };

    struct SanitizeReport {
        // structure - if any of these is non-zero the diagram is NOT rebuilt (it would be meaningless)
        int broken_links {0};   // prev/next mismatch, tau ordering, tau_next != next->tau, tail tau,
                                // or the walk did not reach the tail within max_vertices
        int bad_lines {0};      // conj_vertex / type / w / ph_energy mismatch, internal line ordered
                                // backwards in tau, vertex not (correctly) registered in its manager,
                                // or in the wrong slot of its pair (creation must be even, annihilation odd)

        // momentum: |k_v - (k_prev - sgn(type_v) * w_v)|, the rule every update follows
        double max_momentum_residual {0.};

        // cached state vs what k and tau imply, measured BEFORE the rebuild
        double max_basis_dev {0.};      // |baseWF - U(k)|
        double max_mass_dev {0.};       // relative
        double max_action_dev {0.};     // relative, diagonal of el_prop_action
        double max_overlap_dev {0.};    // |vertex_wf_component - U_prev^T U_v|  (head: vs Identity)
        int overlap_flips {0};          // overlaps off by more than tolerances.flip
        double closing_basis_dev {0.};  // |U(last segment) - U(head)|: the trace closes with no
                                        // overlap, which is exact only if these coincide
        int vertices {0};               // vertices walked, head and tail included

        double trace_before {0.};       // diagram_head->right_component.trace() as found
        double trace_after {0.};        // ... after the rebuild

        // cfg->current_sign disagreed with the sign of the rebuilt trace (it is then repaired)
        bool sign_mismatch {false};

        bool rebuilt {false};

        double relativeTraceChange() const {
            const double scale {std::max(std::abs(trace_before), std::abs(trace_after))};
            return scale > 0. ? std::abs(trace_after - trace_before) / scale : 0.;
        }

        bool clean(const SanitizeTolerances & tol = {}) const {
            return broken_links == 0 && bad_lines == 0 && overlap_flips == 0
                && max_momentum_residual <= tol.momentum
                && max_basis_dev <= tol.exact && max_mass_dev <= tol.exact && max_action_dev <= tol.exact
                && closing_basis_dev <= tol.flip
                && !sign_mismatch
                && relativeTraceChange() <= tol.flip;
        }
    };

    // Rebuilds one segment's cached electronic state from its k and duration. head gets the
    // Identity overlap by convention (it has no predecessor).
    inline void rebuildSegment(Vertex * v, const Vertex * head){
        const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(v->k)};
        v->eff_masses = weight::LKMatrix::computeEffMassfromEigenval(
            std::array<double, 3>{eig(0,0), eig(0,1), eig(0,2)});
        v->baseWF = eig.block<3,3>(1,0);
        v->vertex_wf_component = (v == head)
            ? Eigen::Matrix3d(Eigen::Matrix3d::Identity())
            : Coupling::LKOverlap::computeMatrix(v->prev->baseWF, v->baseWF);
        v->computeElPropAction();
    }

    inline SanitizeReport sanitizeDiagram(diagram_cfg * cfg, const SanitizeTolerances & tol = {}){
        assert(cfg != nullptr);
        SanitizeReport rep;

        Vertex * const head {cfg->diagram_head};
        Vertex * const tail {cfg->diagram_tail};
        rep.trace_before = head->right_component.trace();

        // ---- 1. structure: links, tau ordering, reachability -------------------------------
        if (head->prev != nullptr || tail->next != nullptr) { ++rep.broken_links; }
        if (tail->tau != cfg->current_tau_length) { ++rep.broken_links; }

        int n_internal {0};
        int n_external {0};
        {
            const Vertex * v {head};
            int steps {0};
            while (v != tail && v != nullptr && steps <= cfg->max_vertices) {
                if (v->next == nullptr || v->next->prev != v) { ++rep.broken_links; break; }
                if (!(v->tau < v->tau_next) || v->tau_next != v->next->tau) { ++rep.broken_links; }
                v = v->next;
                ++steps;
            }
            if (v != tail) { ++rep.broken_links; }
            rep.vertices = steps + 1;
        }
        if (rep.broken_links > 0) {
            return rep; // nothing below is meaningful on a broken list
        }

        // ---- 2. lines, manager registration, momentum conservation -------------------------
        for (Vertex * v {head->next}; v != tail; v = v->next) {
            const int c {v->type};
            const bool internal {std::abs(c) == 1};
            if (c == 0 || std::abs(c) > 2) { ++rep.bad_lines; continue; }

            const Vertex * cj {v->conj_vertex};
            if (cj == nullptr || cj->conj_vertex != v || cj->type != -c || cj->w != v->w
                || cj->ph_energy != v->ph_energy) {
                ++rep.bad_lines;
            }
            else if (internal && ((c > 0) != (v->tau < cj->tau))) {
                ++rep.bad_lines; // creation must come first on an internal line
            }

            const VertexPointerManager * m {internal ? cfg->internal_ph_manager : cfg->external_ph_manager};
            (internal ? n_internal : n_external) += 1;
            if (v->index < 0 || v->index >= m->current_length
                || m->ptr_vertex_pool[v->index].linked_vertex != v
                || m->ptr_vertex_pool[v->index].conjugated == nullptr
                || m->ptr_vertex_pool[v->index].conjugated->linked_vertex != cj
                || (c > 0) != (v->index % 2 == 0)) {
                ++rep.bad_lines;
            }

            const double sgn {c > 0 ? 1. : -1.};
            for (int i {0}; i < 3; ++i) {
                const double expected {v->prev->k[i] - sgn * v->w[i]};
                rep.max_momentum_residual = std::max(rep.max_momentum_residual, std::abs(v->k[i] - expected));
            }
        }
        if (n_internal != cfg->internal_ph_manager->current_length) { ++rep.bad_lines; }
        if (n_external != cfg->external_ph_manager->current_length) { ++rep.bad_lines; }

        // ---- 3. cached state vs k and tau, as stored -----------------------------------------
        for (Vertex * v {head}; v != tail; v = v->next) {
            const Eigen::Matrix<double, 4, 3> eig {weight::LKMatrix::diagonalizeLKHamiltonian(v->k)};
            const Eigen::Matrix3d U {eig.block<3,3>(1,0)};
            const std::array<double, 3> m {weight::LKMatrix::computeEffMassfromEigenval(
                std::array<double, 3>{eig(0,0), eig(0,1), eig(0,2)})};

            rep.max_basis_dev = std::max(rep.max_basis_dev, (v->baseWF - U).cwiseAbs().maxCoeff());

            const double k_sq {v->k[0]*v->k[0] + v->k[1]*v->k[1] + v->k[2]*v->k[2]};
            for (int i {0}; i < 3; ++i) {
                rep.max_mass_dev = std::max(rep.max_mass_dev, std::abs(v->eff_masses[i] - m[i]) / std::abs(m[i]));
                const double a {std::exp(-k_sq / (2. * m[i]) * (v->tau_next - v->tau))};
                // floor the denominator: a long, energetic segment underflows to 0, and two zeros agree
                const double denom {std::max(a, std::numeric_limits<double>::min())};
                rep.max_action_dev = std::max(rep.max_action_dev, std::abs(v->el_prop_action(i,i) - a) / denom);
            }

            // overlaps are checked against the STORED neighbouring bases: that is the consistency the
            // trace product needs, and the place where a reused cached overlap shows up
            const Eigen::Matrix3d expected_wf {
                (v == head) ? Eigen::Matrix3d(Eigen::Matrix3d::Identity())
                            : Coupling::LKOverlap::computeMatrix(v->prev->baseWF, v->baseWF)
            };
            const double dev {(v->vertex_wf_component - expected_wf).cwiseAbs().maxCoeff()};
            rep.max_overlap_dev = std::max(rep.max_overlap_dev, dev);
            if (dev > tol.flip) { ++rep.overlap_flips; }
        }
        if (tail->prev != head) {
            rep.closing_basis_dev = (tail->prev->baseWF - head->baseWF).cwiseAbs().maxCoeff();
        }

        // ---- 4. rebuild every cached quantity from k and tau, then refresh the products -----
        for (Vertex * v {head}; v != tail; v = v->next) {
            rebuildSegment(v, head);
        }
        weight::LKMatrix::computeRightSide(head, tail);
        weight::LKMatrix::computeLeftSide(tail, head);
        rep.rebuilt = true;
        rep.trace_after = head->right_component.trace();

        // the sign the updates track is cached state too: check it against the rebuilt trace, and set it
        // (a hand-built diagram starts with the default +1, whatever its trace)
        const int sign_after {rep.trace_after < 0. ? -1 : 1};
        rep.sign_mismatch = cfg->current_sign != sign_after;
        cfg->current_sign = sign_after;

        return rep;
    }

    // Runs sanitizeDiagram every `period` calls of measure() - same call shape as
    // green_trace_measurement, so it can be driven from the same place in the MC loop. Keeps
    // running totals across the whole simulation; in debug builds a dirty diagram aborts.
    struct diagram_sanitizer {
        diagram_cfg * const cfg {nullptr};
        const std::uint64_t period {100000};
        const SanitizeTolerances tol {};

        std::uint64_t calls {0};
        std::uint64_t runs {0};
        std::uint64_t dirty_runs {0};
        std::uint64_t total_flips {0};
        double worst_overlap_dev {0.};
        double worst_trace_change {0.};
        SanitizeReport last {};

        diagram_sanitizer(diagram_cfg * cfg, std::uint64_t period = 100000, SanitizeTolerances tol = {})
            : cfg(cfg), period(period), tol(tol) {
            assert(cfg != nullptr);
            assert(period > 0);
        }

        void measure() {
            if (++calls % period != 0) { return; }
            last = sanitizeDiagram(cfg, tol);
            ++runs;
            total_flips += static_cast<std::uint64_t>(last.overlap_flips);
            worst_overlap_dev = std::max(worst_overlap_dev, last.max_overlap_dev);
            worst_trace_change = std::max(worst_trace_change, last.relativeTraceChange());
            if (!last.clean(tol)) { ++dirty_runs; }
            assert(last.clean(tol) && "diagram_sanitizer: diagram state inconsistent (see `last`)");
        }
    };
}

#endif // !DIAGRAM_SANITIZER_HPP
